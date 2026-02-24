"""
Comprehensive TabFormer Benchmark Sweep

Compares baseline (native model without torch.compile) vs optimized (torch.compile fullgraph).
Tests three model sizes (20M, 120M, 720M) across various sequence lengths and batch sizes.
Computes maximum throughput achievable for SLA targets (5ms, 10ms, 15ms).

Input preparation:
- 14 fields per transaction:
  - 2 categorical fields with 300 categories each
  - 2 categorical fields with 30 categories each
  - 2 numerical fields discretized into 200 buckets each
  - 8 numerical fields discretized into 20 buckets each
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse

# Import native GPT and field transformer
from models.gpt_native import NativeGPTLMHeadModel
from models.hierarchical import TabFormerEmbeddings

# Increase torch.compile cache size limit to handle multiple (batch_size, seq_len) shapes
torch._dynamo.config.cache_size_limit = 128  # Default is 64


@dataclass
class TabFormerConfig:
    """TabFormer configuration."""
    # Model identifier
    name: str

    # Field transformer config
    field_hidden_size: int  # Internal embedding size / num_fields
    field_num_layers: int
    field_nhead: int
    field_ffn_size: int

    # Sequence transformer (GPT) config
    hidden_size: int  # Internal embedding size
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int

    # Output config
    final_embedding_size: int

    # Common config
    ncols: int = 14  # Number of fields
    vocab_size: int = 300 + 2  # Max vocab + padding + special tokens
    max_position_embeddings: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0

    # Enable native implementations
    native_field_transformer: bool = True

    @property
    def estimated_params(self):
        """Estimate total parameters."""
        # Field embeddings
        field_emb = self.vocab_size * self.field_hidden_size * self.ncols

        # Field transformer
        field_attn = self.field_num_layers * (4 * self.field_hidden_size ** 2)
        field_ffn = self.field_num_layers * (2 * self.field_hidden_size * self.field_ffn_size)

        # Projection
        proj = self.field_hidden_size * self.ncols * self.hidden_size

        # Sequence transformer (GPT)
        seq_attn = self.num_hidden_layers * (4 * self.hidden_size ** 2)
        seq_ffn = self.num_hidden_layers * (2 * self.hidden_size * self.intermediate_size)

        # Output
        output = self.hidden_size * self.final_embedding_size

        total = field_emb + field_attn + field_ffn + proj + seq_attn + seq_ffn + output
        return total / 1e6  # Convert to millions


def get_model_configs():
    """Get the three model configurations from the table."""
    configs = {
        '20M': TabFormerConfig(
            name='20M',
            field_hidden_size=384,  # Same as hidden_size, divisible by field_nhead (12)
            field_num_layers=4,
            field_nhead=12,
            field_ffn_size=768,
            hidden_size=384,
            num_hidden_layers=8,
            num_attention_heads=12,
            intermediate_size=768,
            final_embedding_size=256,
        ),
        '120M': TabFormerConfig(
            name='120M',
            field_hidden_size=768,  # Same as hidden_size, divisible by field_nhead (12)
            field_num_layers=6,
            field_nhead=12,
            field_ffn_size=2048,
            hidden_size=768,
            num_hidden_layers=10,
            num_attention_heads=12,
            intermediate_size=2048,
            final_embedding_size=512,
        ),
        '720M': TabFormerConfig(
            name='720M',
            field_hidden_size=1536,  # Same as hidden_size, divisible by field_nhead (24)
            field_num_layers=6,
            field_nhead=24,
            field_ffn_size=4096,
            hidden_size=1536,
            num_hidden_layers=18,
            num_attention_heads=24,
            intermediate_size=4096,
            final_embedding_size=1024,
        ),
    }
    return configs


class TabFormerNative(nn.Module):
    """
    Complete TabFormer model with native field and sequence transformers.

    Architecture:
    1. Field embeddings (14 separate embedding layers)
    2. Field transformer (bidirectional)
    3. Concatenate and project
    4. Sequence transformer (causal GPT)
    5. Output projection
    """
    def __init__(self, config: TabFormerConfig):
        super().__init__()
        self.config = config

        # Field embeddings - separate for each field type
        # Field 0-1: 300 categories
        self.field_emb_0 = nn.Embedding(300 + 2, config.field_hidden_size, padding_idx=0)
        self.field_emb_1 = nn.Embedding(300 + 2, config.field_hidden_size, padding_idx=0)

        # Field 2-3: 30 categories
        self.field_emb_2 = nn.Embedding(30 + 2, config.field_hidden_size, padding_idx=0)
        self.field_emb_3 = nn.Embedding(30 + 2, config.field_hidden_size, padding_idx=0)

        # Field 4-5: 200 buckets (numerical)
        self.field_emb_4 = nn.Embedding(200 + 2, config.field_hidden_size, padding_idx=0)
        self.field_emb_5 = nn.Embedding(200 + 2, config.field_hidden_size, padding_idx=0)

        # Field 6-13: 20 buckets each (numerical)
        self.field_embs_6_13 = nn.ModuleList([
            nn.Embedding(20 + 2, config.field_hidden_size, padding_idx=0)
            for _ in range(8)
        ])

        # Field transformer (bidirectional)
        from models.hierarchical import NativeFieldTransformerEncoder
        self.field_transformer = NativeFieldTransformerEncoder(
            d_model=config.field_hidden_size,
            nhead=config.field_nhead,
            num_layers=config.field_num_layers,
            dim_feedforward=config.field_ffn_size,
            dropout=config.hidden_dropout_prob,
        )

        # Projection from concatenated fields to hidden size
        self.field_projection = nn.Linear(
            config.field_hidden_size * config.ncols,
            config.hidden_size
        )

        # Sequence transformer (causal GPT)
        self.sequence_transformer = NativeGPTLMHeadModel(config)

        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.final_embedding_size
        )

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len, ncols) - field values
        Returns:
            embeddings: (batch, seq_len, final_embedding_size)
        """
        batch_size, seq_len, ncols = input_ids.shape
        assert ncols == 14, f"Expected 14 fields, got {ncols}"

        # Embed each field separately
        field_embeds = []
        field_embeds.append(self.field_emb_0(input_ids[..., 0]))
        field_embeds.append(self.field_emb_1(input_ids[..., 1]))
        field_embeds.append(self.field_emb_2(input_ids[..., 2]))
        field_embeds.append(self.field_emb_3(input_ids[..., 3]))
        field_embeds.append(self.field_emb_4(input_ids[..., 4]))
        field_embeds.append(self.field_emb_5(input_ids[..., 5]))
        for i, emb_layer in enumerate(self.field_embs_6_13):
            field_embeds.append(emb_layer(input_ids[..., 6 + i]))

        # Stack: (batch, seq_len, ncols, field_hidden_size)
        field_embeds = torch.stack(field_embeds, dim=2)

        # Reshape for field transformer: (batch*seq_len, ncols, field_hidden_size)
        field_embeds_flat = field_embeds.view(batch_size * seq_len, ncols, self.config.field_hidden_size)

        # Transpose to (ncols, batch*seq_len, field_hidden_size) for transformer
        field_embeds_flat = field_embeds_flat.transpose(0, 1)

        # Apply field transformer (bidirectional attention across fields)
        field_output = self.field_transformer(field_embeds_flat)

        # Transpose back and reshape: (batch*seq_len, ncols, field_hidden_size)
        field_output = field_output.transpose(0, 1)

        # Reshape to (batch, seq_len, ncols, field_hidden_size)
        field_output = field_output.view(batch_size, seq_len, ncols, self.config.field_hidden_size)

        # Concatenate fields and project
        field_concat = field_output.view(batch_size, seq_len, -1)
        sequence_input = self.field_projection(field_concat)

        # Apply sequence transformer (causal GPT)
        # Note: NativeGPTLMHeadModel expects input_ids, but we'll pass embeddings
        # Need to use the underlying transformer
        transformer_output = self.sequence_transformer.transformer(
            inputs_embeds=sequence_input
        )
        hidden_states = transformer_output[0]

        # Output projection
        output = self.output_projection(hidden_states)

        return output


def create_sample_input(batch_size: int, seq_len: int, device: str = 'cuda'):
    """
    Create sample input matching the field specification.

    Returns:
        input_ids: (batch, seq_len, 14) tensor with appropriate ranges per field
    """
    input_ids = torch.zeros(batch_size, seq_len, 14, dtype=torch.long, device=device)

    # Field 0-1: 300 categories (1-300, 0 is padding)
    input_ids[..., 0] = torch.randint(1, 301, (batch_size, seq_len), device=device)
    input_ids[..., 1] = torch.randint(1, 301, (batch_size, seq_len), device=device)

    # Field 2-3: 30 categories
    input_ids[..., 2] = torch.randint(1, 31, (batch_size, seq_len), device=device)
    input_ids[..., 3] = torch.randint(1, 31, (batch_size, seq_len), device=device)

    # Field 4-5: 200 buckets
    input_ids[..., 4] = torch.randint(1, 201, (batch_size, seq_len), device=device)
    input_ids[..., 5] = torch.randint(1, 201, (batch_size, seq_len), device=device)

    # Field 6-13: 20 buckets each
    for i in range(8):
        input_ids[..., 6 + i] = torch.randint(1, 21, (batch_size, seq_len), device=device)

    return input_ids


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    warmup_iters: int = 5,
    bench_iters: int = 50,
    device: str = 'cuda',
    use_bf16: bool = False,
) -> Dict:
    """
    Benchmark model throughput.

    Args:
        use_bf16: If True, convert model parameters to BF16
                  If False, use full FP32 precision

    FP32 mode:
    - Linear layers: FP32
    - Embeddings: FP32
    - RMSNorm: FP32
    - Softmax: FP32
    - RoPE: FP32 cos/sin tables

    BF16 mode (model.bfloat16()):
    - Linear layers: BF16 parameters
    - Embeddings: BF16 parameters
    - RMSNorm: FP32 computation (enforced in implementation)
    - Softmax: FP32 (scaled_dot_product_attention handles this)
    - RoPE: FP32 cos/sin tables (enforced via _apply override)

    Returns dict with:
        - latency_ms: mean latency in milliseconds
        - throughput: sequences per second
        - memory_mb: peak memory in MB
    """
    model.eval()

    # Convert model dtype once at the beginning
    if use_bf16:
        model = model.bfloat16()
    else:
        model = model.float()

    # Create input
    input_ids = create_sample_input(batch_size, seq_len, device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids)

    # Synchronize before benchmarking
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(bench_iters):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(input_ids)

            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

    # Get memory stats
    if device == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0

    mean_latency = np.mean(latencies)
    throughput = (batch_size / mean_latency) * 1000  # sequences/sec

    return {
        'latency_ms': mean_latency,
        'latency_std_ms': np.std(latencies),
        'throughput': throughput,
        'memory_mb': memory_mb,
    }


def find_max_batch_size_for_sla(
    config: TabFormerConfig,
    seq_len: int,
    sla_ms: float,
    device: str = 'cuda',
    max_batch_size: int = 256,
    use_bf16: bool = False,
    use_compile: bool = False,
) -> Dict:
    """
    Find maximum batch size that meets SLA.

    Strategy:
    - Double batch size until 32: 1, 2, 4, 8, 16, 32
    - Then increment by 16: 48, 64, 80, 96, ...
    - Recompile model for each batch_size for proper CUDA graph capture

    Args:
        config: Model configuration
        use_compile: If True, compile model with torch.compile(fullgraph=True, mode='reduce-overhead')

    Returns dict with:
        - max_batch_size: Largest batch size meeting SLA
        - max_throughput: Throughput at that batch size
        - latency_ms: Latency at that batch size
        - all_results: List of all (batch_size, latency, throughput) tested
    """
    all_results = []
    best_batch_size = 1
    best_throughput = 0
    best_latency = float('inf')

    batch_size = 1

    while batch_size <= max_batch_size:
        try:
            # Create fresh model for each batch_size to enable CUDA graph capture
            model = TabFormerNative(config).to(device)
            if use_compile:
                model = torch.compile(
                    model,
                    mode='reduce-overhead',
                    fullgraph=True
                )
            model.eval()

            metrics = benchmark_model(
                model,
                batch_size,
                seq_len,
                warmup_iters=20,  # Increased for CUDA graph capture
                bench_iters=50,
                device=device,
                use_bf16=use_bf16,
            )

            # Clean up model to free memory
            del model
            torch.cuda.empty_cache()

            latency = metrics['latency_ms']
            throughput = metrics['throughput']

            all_results.append({
                'batch_size': batch_size,
                'latency_ms': latency,
                'throughput': throughput,
                'memory_mb': metrics['memory_mb'],
            })

            print(f"      bs={batch_size}: {latency:.2f}ms, {throughput:.1f} seq/s", end="")

            if latency <= sla_ms:
                # Still within SLA, update best
                best_batch_size = batch_size
                best_throughput = throughput
                best_latency = latency
                print(f" ✓")

                # Determine next batch size
                if batch_size < 32:
                    batch_size *= 2  # Double until 32
                else:
                    batch_size += 16  # Increment by 16 after 32
            else:
                # Exceeded SLA, stop searching
                print(f" ✗ (exceeds {sla_ms}ms SLA)")
                break

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"      bs={batch_size}: OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise

    return {
        'max_batch_size': best_batch_size,
        'max_throughput': best_throughput,
        'latency_ms': best_latency,
        'all_results': all_results,
    }


def extract_sla_results(all_results: List[Dict], sla_targets_ms: List[float]) -> Dict:
    """
    Extract best batch size for each SLA from collected results.

    Args:
        all_results: List of dicts with batch_size, latency_ms, throughput
        sla_targets_ms: List of SLA targets to extract

    Returns:
        Dict mapping sla_ms -> {max_batch_size, max_throughput, latency_ms}
    """
    sla_results = {}

    for sla_ms in sla_targets_ms:
        best_batch_size = 1
        best_throughput = 0
        best_latency = float('inf')

        for result in all_results:
            if result['latency_ms'] <= sla_ms:
                if result['throughput'] > best_throughput:
                    best_batch_size = result['batch_size']
                    best_throughput = result['throughput']
                    best_latency = result['latency_ms']

        sla_results[sla_ms] = {
            'max_batch_size': best_batch_size,
            'max_throughput': best_throughput,
            'latency_ms': best_latency,
        }

    return sla_results


def run_sweep(
    config_name: str,
    config: TabFormerConfig,
    seq_lengths: List[int],
    sla_targets_ms: List[float],
    device: str = 'cuda',
    use_bf16: bool = False,
) -> Dict:
    """Run SLA-based sweep for a model config."""
    print(f"\n{'='*80}")
    print(f"Running sweep for {config_name} TabFormer")
    print(f"Estimated params: {config.estimated_params:.1f}M")
    print(f"{'='*80}")

    # Use maximum SLA for actual benchmarking
    max_sla = max(sla_targets_ms)
    print(f"Running with max SLA: {max_sla}ms, will extract results for all SLA targets")

    results = {
        'config_name': config_name,
        'estimated_params_m': config.estimated_params,
        'baseline': {},
        'optimized': {},
    }

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        results['baseline'][seq_len] = {}
        results['optimized'][seq_len] = {}

        try:
            # Baseline: Native model without compile
            print(f"  Baseline (no compile):")
            baseline_results = find_max_batch_size_for_sla(
                config,
                seq_len,
                max_sla,
                device=device,
                use_bf16=use_bf16,
                use_compile=False,
            )

            # Extract results for all SLA targets
            baseline_sla_results = extract_sla_results(
                baseline_results['all_results'],
                sla_targets_ms
            )

            for sla_ms, sla_data in baseline_sla_results.items():
                results['baseline'][seq_len][sla_ms] = sla_data
                print(f"    SLA {sla_ms}ms: bs={sla_data['max_batch_size']}, "
                      f"{sla_data['max_throughput']:.1f} seq/s @ {sla_data['latency_ms']:.2f}ms")

            # Optimized: With torch.compile fullgraph (recompiles for each batch_size)
            print(f"  Optimized (torch.compile, recompile per config for CUDA graphs):")
            optimized_results = find_max_batch_size_for_sla(
                config,
                seq_len,
                max_sla,
                device=device,
                use_bf16=use_bf16,
                use_compile=True,
            )

            # Extract results for all SLA targets
            optimized_sla_results = extract_sla_results(
                optimized_results['all_results'],
                sla_targets_ms
            )

            for sla_ms, sla_data in optimized_sla_results.items():
                results['optimized'][seq_len][sla_ms] = sla_data

                baseline_throughput = baseline_sla_results[sla_ms]['max_throughput']
                if baseline_throughput > 0:
                    speedup = sla_data['max_throughput'] / baseline_throughput
                else:
                    speedup = 0

                print(f"    SLA {sla_ms}ms: bs={sla_data['max_batch_size']}, "
                      f"{sla_data['max_throughput']:.1f} seq/s @ {sla_data['latency_ms']:.2f}ms "
                      f"({speedup:.2f}x)")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            for sla_ms in sla_targets_ms:
                results['baseline'][seq_len][sla_ms] = None
                results['optimized'][seq_len][sla_ms] = None
            torch.cuda.empty_cache()

    return results


def save_results(all_results: Dict, output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def print_summary(all_results: Dict):
    """Print summary of results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for config_name, results in all_results.items():
        print(f"\n{config_name} TabFormer ({results['estimated_params_m']:.1f}M parameters)")
        print("-" * 80)

        # Get unique seq lengths and SLA targets
        seq_lengths = sorted(results['baseline'].keys())

        for seq_len in seq_lengths:
            print(f"\nSequence Length: {seq_len}")
            print(f"  {'SLA':<10} {'Baseline':<35} {'Optimized':<35} {'Speedup':<10}")
            print(f"  {'-'*10} {'-'*35} {'-'*35} {'-'*10}")

            sla_targets = sorted(results['baseline'][seq_len].keys())
            for sla_ms in sla_targets:
                baseline_data = results['baseline'][seq_len][sla_ms]
                optimized_data = results['optimized'][seq_len][sla_ms]

                if baseline_data and optimized_data:
                    baseline_str = f"{baseline_data['max_throughput']:.1f} seq/s (bs={baseline_data['max_batch_size']})"
                    optimized_str = f"{optimized_data['max_throughput']:.1f} seq/s (bs={optimized_data['max_batch_size']})"

                    if baseline_data['max_throughput'] > 0 and optimized_data['max_throughput'] > 0:
                        speedup = optimized_data['max_throughput'] / baseline_data['max_throughput']
                        speedup_str = f"{speedup:.2f}x"
                    else:
                        speedup_str = "N/A"
                else:
                    baseline_str = "N/A"
                    optimized_str = "N/A"
                    speedup_str = "N/A"

                print(f"  {sla_ms:<10.1f} {baseline_str:<35} {optimized_str:<35} {speedup_str:<10}")


def main():
    parser = argparse.ArgumentParser(description='TabFormer Benchmark Sweep')
    parser.add_argument('--configs', nargs='+', default=['20M', '120M', '720M'],
                        choices=['20M', '120M', '720M'],
                        help='Model configs to test')
    parser.add_argument('--seq-lengths', nargs='+', type=int,
                        default=[32, 64, 128, 256, 512, 1024],
                        help='Sequence lengths to test')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output', default='tabformer_sweep_results.json',
                        help='Output JSON file')
    parser.add_argument('--sla-targets', nargs='+', type=float,
                        default=[5.0, 10.0, 15.0],
                        help='SLA targets in milliseconds')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32',
                        help='Precision mode: fp32 (full precision) or bf16 (mixed precision with autocast)')

    args = parser.parse_args()

    use_bf16 = (args.precision == 'bf16')

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("="*80)
    print(f"TABFORMER BENCHMARK SWEEP - {args.precision.upper()} PRECISION")
    print("="*80)
    print(f"Device: {args.device}")
    if use_bf16:
        print(f"Precision: BF16 (mixed precision with autocast)")
        print(f"  - Linear/Embeddings: BF16")
        print(f"  - RMSNorm/Softmax/RoPE: FP32")
    else:
        print(f"Precision: FP32 (full precision)")
        print(f"  - All operations: FP32")
    print(f"Configs: {args.configs}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"SLA targets: {args.sla_targets}ms")
    print(f"Batch size strategy: Double until 32, then +16 (1,2,4,8,16,32,48,64...)")
    print(f"Warmup: 5 iterations, Benchmark: 50 iterations")

    # Get model configs
    all_model_configs = get_model_configs()

    # Run sweeps
    all_results = {}
    for config_name in args.configs:
        if config_name not in all_model_configs:
            print(f"Warning: Unknown config {config_name}, skipping")
            continue

        config = all_model_configs[config_name]
        results = run_sweep(
            config_name,
            config,
            args.seq_lengths,
            args.sla_targets,
            args.device,
            use_bf16,
        )

        all_results[config_name] = results

    # Save results
    save_results(all_results, args.output)

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
