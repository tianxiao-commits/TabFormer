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
            field_hidden_size=384 // 14,  # 384 / 14 fields ≈ 27
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
            field_hidden_size=768 // 14,  # 768 / 14 ≈ 54
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
            field_hidden_size=1536 // 14,  # 1536 / 14 ≈ 109
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
    warmup_iters: int = 100,
    bench_iters: int = 100,
    device: str = 'cuda'
) -> Dict:
    """
    Benchmark model throughput.

    Returns dict with:
        - latency_ms: mean latency in milliseconds
        - throughput: sequences per second
        - memory_mb: peak memory in MB
    """
    model.eval()

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


def run_sweep(
    config_name: str,
    config: TabFormerConfig,
    seq_lengths: List[int],
    batch_sizes: List[int],
    device: str = 'cuda',
) -> Dict:
    """Run full sweep for a model config."""
    print(f"\n{'='*80}")
    print(f"Running sweep for {config_name} TabFormer")
    print(f"Estimated params: {config.estimated_params:.1f}M")
    print(f"{'='*80}")

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

        for batch_size in batch_sizes:
            print(f"  Batch Size: {batch_size}")

            try:
                # Baseline: Native model without compile
                model_baseline = TabFormerNative(config).to(device)
                model_baseline.eval()

                baseline_metrics = benchmark_model(
                    model_baseline,
                    batch_size,
                    seq_len,
                    warmup_iters=100,
                    bench_iters=100,
                    device=device
                )

                results['baseline'][seq_len][batch_size] = baseline_metrics
                print(f"    Baseline - Latency: {baseline_metrics['latency_ms']:.2f}ms, "
                      f"Throughput: {baseline_metrics['throughput']:.1f} seq/s, "
                      f"Memory: {baseline_metrics['memory_mb']:.0f}MB")

                # Clean up
                del model_baseline
                torch.cuda.empty_cache()

                # Optimized: With torch.compile fullgraph
                model_optimized = TabFormerNative(config).to(device)
                model_optimized = torch.compile(
                    model_optimized,
                    mode='reduce-overhead',
                    fullgraph=True
                )
                model_optimized.eval()

                optimized_metrics = benchmark_model(
                    model_optimized,
                    batch_size,
                    seq_len,
                    warmup_iters=100,  # Warmup for CUDA graph capture
                    bench_iters=100,
                    device=device
                )

                results['optimized'][seq_len][batch_size] = optimized_metrics
                speedup = baseline_metrics['latency_ms'] / optimized_metrics['latency_ms']
                print(f"    Optimized - Latency: {optimized_metrics['latency_ms']:.2f}ms, "
                      f"Throughput: {optimized_metrics['throughput']:.1f} seq/s, "
                      f"Memory: {optimized_metrics['memory_mb']:.0f}MB, "
                      f"Speedup: {speedup:.2f}x")

                # Clean up
                del model_optimized
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    OOM - skipping")
                    results['baseline'][seq_len][batch_size] = None
                    results['optimized'][seq_len][batch_size] = None
                    torch.cuda.empty_cache()
                else:
                    raise

    return results


def compute_sla_throughput(results: Dict, sla_targets_ms: List[float]) -> Dict:
    """
    Compute maximum throughput achievable for each SLA target.

    For each SLA target (e.g., 5ms), find the maximum batch size that meets
    the latency requirement, then compute throughput.
    """
    sla_results = {
        'baseline': {sla: {} for sla in sla_targets_ms},
        'optimized': {sla: {} for sla in sla_targets_ms},
    }

    for mode in ['baseline', 'optimized']:
        for seq_len, batch_results in results[mode].items():
            for sla_ms in sla_targets_ms:
                max_throughput = 0
                best_batch_size = None

                for batch_size, metrics in batch_results.items():
                    if metrics is None:
                        continue

                    if metrics['latency_ms'] <= sla_ms:
                        if metrics['throughput'] > max_throughput:
                            max_throughput = metrics['throughput']
                            best_batch_size = batch_size

                sla_results[mode][sla_ms][seq_len] = {
                    'max_throughput': max_throughput,
                    'best_batch_size': best_batch_size,
                }

    return sla_results


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

        # Print SLA throughput comparison
        sla_results = results['sla_throughput']

        for sla_ms in [5, 10, 15]:
            print(f"\nSLA: {sla_ms}ms")
            print(f"  {'Seq Len':<10} {'Baseline':<25} {'Optimized':<25} {'Speedup':<10}")
            print(f"  {'-'*10} {'-'*25} {'-'*25} {'-'*10}")

            for seq_len in sorted(sla_results['baseline'][sla_ms].keys()):
                baseline_data = sla_results['baseline'][sla_ms][seq_len]
                optimized_data = sla_results['optimized'][sla_ms][seq_len]

                baseline_str = f"{baseline_data['max_throughput']:.1f} seq/s (bs={baseline_data['best_batch_size']})"
                optimized_str = f"{optimized_data['max_throughput']:.1f} seq/s (bs={optimized_data['best_batch_size']})"

                if baseline_data['max_throughput'] > 0 and optimized_data['max_throughput'] > 0:
                    speedup = optimized_data['max_throughput'] / baseline_data['max_throughput']
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"

                print(f"  {seq_len:<10} {baseline_str:<25} {optimized_str:<25} {speedup_str:<10}")


def main():
    parser = argparse.ArgumentParser(description='TabFormer Benchmark Sweep')
    parser.add_argument('--configs', nargs='+', default=['20M', '120M', '720M'],
                        choices=['20M', '120M', '720M'],
                        help='Model configs to test')
    parser.add_argument('--seq-lengths', nargs='+', type=int,
                        default=[32, 64, 128, 256, 512, 1024],
                        help='Sequence lengths to test')
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                        default=[1, 2, 4, 8, 16, 32],
                        help='Batch sizes to test')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output', default='tabformer_sweep_results.json',
                        help='Output JSON file')
    parser.add_argument('--sla-targets', nargs='+', type=float,
                        default=[5.0, 10.0, 15.0],
                        help='SLA targets in milliseconds')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("="*80)
    print("TABFORMER BENCHMARK SWEEP")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Configs: {args.configs}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"SLA targets: {args.sla_targets}ms")

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
            args.batch_sizes,
            args.device
        )

        # Compute SLA throughput
        sla_throughput = compute_sla_throughput(results, args.sla_targets)
        results['sla_throughput'] = sla_throughput

        all_results[config_name] = results

    # Save results
    save_results(all_results, args.output)

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
