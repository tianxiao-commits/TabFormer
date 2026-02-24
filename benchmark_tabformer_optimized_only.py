"""
TabFormer Optimized Benchmark Sweep

Tests torch.compile(fullgraph=True) optimized version with configurable precision.
Supports both BF16 and FP32 precision via --precision flag.
Uses binary search to find maximum batch size for each SLA target (5ms, 10ms, 15ms).
Tests three model sizes (20M, 120M, 720M) across various sequence lengths.

Features:
- Always measures batch_size=1 as baseline
- Binary search from 2-128 for each SLA target
- Per-config recompilation for CUDA graph capture
- Tracks mean, median, and std of latency across 50 iterations

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
            num_hidden_layers=4,
            num_attention_heads=12,
            intermediate_size=768,
            final_embedding_size=384,
        ),
        '120M': TabFormerConfig(
            name='120M',
            field_hidden_size=768,
            field_num_layers=6,
            field_nhead=12,
            field_ffn_size=1536,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=1536,
            final_embedding_size=768,
        ),
        '720M': TabFormerConfig(
            name='720M',
            field_hidden_size=1536,
            field_num_layers=12,
            field_nhead=24,
            field_ffn_size=3072,
            hidden_size=1536,
            num_hidden_layers=12,
            num_attention_heads=24,
            intermediate_size=3072,
            final_embedding_size=1536,
        ),
    }
    return configs


class TabFormerNative(nn.Module):
    """
    TabFormer with native PyTorch implementations.

    Architecture:
    1. Field-level embeddings: Each of 14 fields gets its own embedding table
    2. Field transformer: Bidirectional attention across fields (ncols dimension)
    3. Field projection: Concatenate fields and project to sequence hidden size
    4. Sequence transformer: Causal GPT with RoPE and RMSNorm
    5. Output projection: Project to final embedding size
    """
    def __init__(self, config: TabFormerConfig):
        super().__init__()
        self.config = config

        # Field embeddings: Separate embedding table for each field
        # Fields 0-1: 300 categories
        self.field_emb_0 = nn.Embedding(config.vocab_size, config.field_hidden_size, padding_idx=config.pad_token_id)
        self.field_emb_1 = nn.Embedding(config.vocab_size, config.field_hidden_size, padding_idx=config.pad_token_id)

        # Fields 2-3: 30 categories (smaller vocab, but same embedding size)
        self.field_emb_2 = nn.Embedding(config.vocab_size, config.field_hidden_size, padding_idx=config.pad_token_id)
        self.field_emb_3 = nn.Embedding(config.vocab_size, config.field_hidden_size, padding_idx=config.pad_token_id)

        # Fields 4-5: 200 buckets
        self.field_emb_4 = nn.Embedding(config.vocab_size, config.field_hidden_size, padding_idx=config.pad_token_id)
        self.field_emb_5 = nn.Embedding(config.vocab_size, config.field_hidden_size, padding_idx=config.pad_token_id)

        # Fields 6-13: 20 buckets each
        self.field_embs_6_13 = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.field_hidden_size, padding_idx=config.pad_token_id)
            for _ in range(8)
        ])

        # Field transformer (bidirectional attention across fields)
        from models.hierarchical import NativeFieldTransformerEncoder
        self.field_transformer = NativeFieldTransformerEncoder(
            d_model=config.field_hidden_size,
            nhead=config.field_nhead,
            num_layers=config.field_num_layers,
            dim_feedforward=config.field_ffn_size,
            dropout=config.hidden_dropout_prob,
        )

        # Project concatenated fields to sequence hidden size
        self.field_projection = nn.Linear(
            config.field_hidden_size * config.ncols,
            config.hidden_size
        )

        # Sequence transformer (causal GPT)
        self.sequence_transformer = NativeGPTLMHeadModel(config)

        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.final_embedding_size)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len, ncols) where ncols=14
        Returns:
            output: (batch, seq_len, final_embedding_size)
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
    warmup_iters: int = 20,
    bench_iters: int = 50,
    device: str = 'cuda',
    use_bf16: bool = True,
) -> Dict:
    """
    Benchmark model throughput with BF16 precision.

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

    # Convert model to BF16
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
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    throughput = (batch_size / mean_latency) * 1000  # sequences/sec

    return {
        'latency_ms': mean_latency,
        'latency_median_ms': median_latency,
        'latency_std_ms': std_latency,
        'throughput': throughput,
        'memory_mb': memory_mb,
    }


def benchmark_batch_size(
    config: TabFormerConfig,
    batch_size: int,
    seq_len: int,
    device: str = 'cuda',
    use_bf16: bool = True,
) -> Dict:
    """
    Benchmark a specific batch size with torch.compile.

    Returns metrics dict or None if OOM.
    """
    try:
        # Create fresh model for each batch_size to enable CUDA graph capture
        model = TabFormerNative(config).to(device)
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

        return metrics

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return None
        else:
            raise


def find_max_batch_size_for_sla(
    config: TabFormerConfig,
    seq_len: int,
    sla_targets_ms: List[float],
    device: str = 'cuda',
    max_batch_size: int = 128,
    use_bf16: bool = True,
) -> Dict:
    """
    Find maximum batch size for each SLA target using binary search.

    Strategy:
    1. Always measure batch_size=1
    2. For each SLA target, binary search from 2 to 128

    Returns dict with results for each SLA target and all measurements.
    """
    all_measurements = {}

    # Always measure batch_size=1
    print(f"    Measuring batch_size=1 (reference point)...")
    metrics = benchmark_batch_size(config, 1, seq_len, device, use_bf16)

    if metrics is None:
        print(f"      OOM at batch_size=1!")
        return {'measurements': {}, 'sla_results': {}}

    all_measurements[1] = {
        'batch_size': 1,
        'latency_ms': metrics['latency_ms'],
        'latency_median_ms': metrics['latency_median_ms'],
        'latency_std_ms': metrics['latency_std_ms'],
        'throughput': metrics['throughput'],
        'memory_mb': metrics['memory_mb'],
    }

    print(f"      bs=1: {metrics['latency_ms']:.2f}ms (median: {metrics['latency_median_ms']:.2f}ms, "
          f"std: {metrics['latency_std_ms']:.2f}ms), {metrics['throughput']:.1f} seq/s")

    # Binary search for each SLA target
    sla_results = {}

    for sla_ms in sorted(sla_targets_ms):
        print(f"\n    Binary search for SLA {sla_ms}ms (max_bs=128)...")

        # Check if batch_size=1 already exceeds SLA
        if all_measurements[1]['latency_ms'] > sla_ms:
            print(f"      bs=1 already exceeds SLA ({all_measurements[1]['latency_ms']:.2f}ms > {sla_ms}ms)")
            sla_results[sla_ms] = {
                'sla_met': False,
                'max_batch_size': 1,
                'latency_ms': all_measurements[1]['latency_ms'],
                'latency_median_ms': all_measurements[1]['latency_median_ms'],
                'latency_std_ms': all_measurements[1]['latency_std_ms'],
                'throughput': all_measurements[1]['throughput'],
            }
            continue

        # Binary search from 2 to max_batch_size
        left, right = 2, max_batch_size
        best_bs = 1
        best_metrics = all_measurements[1]

        while left <= right:
            mid = (left + right) // 2

            # Check if already measured
            if mid not in all_measurements:
                print(f"      Testing bs={mid}...")
                metrics = benchmark_batch_size(config, mid, seq_len, device, use_bf16)

                if metrics is None:
                    print(f"        OOM at bs={mid}, searching lower")
                    right = mid - 1
                    continue

                all_measurements[mid] = {
                    'batch_size': mid,
                    'latency_ms': metrics['latency_ms'],
                    'latency_median_ms': metrics['latency_median_ms'],
                    'latency_std_ms': metrics['latency_std_ms'],
                    'throughput': metrics['throughput'],
                    'memory_mb': metrics['memory_mb'],
                }

                print(f"        bs={mid}: {metrics['latency_ms']:.2f}ms (median: {metrics['latency_median_ms']:.2f}ms, "
                      f"std: {metrics['latency_std_ms']:.2f}ms), {metrics['throughput']:.1f} seq/s")

            metrics_dict = all_measurements[mid]

            if metrics_dict['latency_ms'] <= sla_ms:
                # Within SLA, try larger batch size
                best_bs = mid
                best_metrics = metrics_dict
                left = mid + 1
            else:
                # Exceeds SLA, try smaller batch size
                right = mid - 1

        sla_results[sla_ms] = {
            'sla_met': True,
            'max_batch_size': best_bs,
            'latency_ms': best_metrics['latency_ms'],
            'latency_median_ms': best_metrics['latency_median_ms'],
            'latency_std_ms': best_metrics['latency_std_ms'],
            'throughput': best_metrics['throughput'],
        }

        print(f"      Best: bs={best_bs}, {best_metrics['latency_ms']:.2f}ms, {best_metrics['throughput']:.1f} seq/s")

    return {
        'measurements': all_measurements,
        'sla_results': sla_results,
    }


def extract_sla_results(all_results: List[Dict], sla_targets_ms: List[float]) -> Dict:
    """
    Extract best batch size for each SLA from collected results.

    Args:
        all_results: List of dicts with batch_size, latency_ms, throughput
        sla_targets_ms: List of SLA targets to extract

    Returns:
        Dict mapping sla_ms -> {max_batch_size, max_throughput, latency_ms, sla_met, bs1_throughput, bs1_latency}
    """
    sla_results = {}

    # Get bs=1 baseline data
    bs1_result = next((r for r in all_results if r['batch_size'] == 1), None)
    bs1_throughput = bs1_result['throughput'] if bs1_result else 0
    bs1_latency = bs1_result['latency_ms'] if bs1_result else float('inf')

    for sla_ms in sla_targets_ms:
        best_batch_size = 1
        best_throughput = 0
        best_latency = float('inf')
        sla_met = False

        for result in all_results:
            if result['latency_ms'] <= sla_ms:
                sla_met = True
                if result['throughput'] > best_throughput:
                    best_batch_size = result['batch_size']
                    best_throughput = result['throughput']
                    best_latency = result['latency_ms']

        # If SLA not met, still include bs=1 data for analysis
        if not sla_met and bs1_result:
            best_batch_size = 1
            best_throughput = bs1_throughput
            best_latency = bs1_latency

        sla_results[sla_ms] = {
            'max_batch_size': best_batch_size,
            'max_throughput': best_throughput,
            'latency_ms': best_latency,
            'sla_met': sla_met,
            'bs1_throughput': bs1_throughput,
            'bs1_latency': bs1_latency,
        }

    return sla_results


def run_sweep(
    config_name: str,
    config: TabFormerConfig,
    seq_lengths: List[int],
    sla_targets_ms: List[float],
    device: str = 'cuda',
    use_bf16: bool = True,
) -> Dict:
    """Run SLA-based sweep for a model config (optimized only)."""
    print(f"\n{'='*80}")
    print(f"Running optimized sweep for {config_name} TabFormer")
    print(f"Estimated params: {config.estimated_params:.1f}M")
    print(f"{'='*80}")

    # Use maximum SLA for actual benchmarking
    max_sla = max(sla_targets_ms)
    print(f"Running with max SLA: {max_sla}ms, will extract results for all SLA targets")

    results = {
        'config_name': config_name,
        'estimated_params_m': config.estimated_params,
        'optimized': {},
    }

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        results['optimized'][seq_len] = {}

        try:
            # Optimized: With torch.compile fullgraph (binary search for each SLA)
            print(f"  Optimized (torch.compile + CUDA graphs, BF16):")
            search_results = find_max_batch_size_for_sla(
                config,
                seq_len,
                sla_targets_ms,
                device=device,
                max_batch_size=128,
                use_bf16=use_bf16,
            )

            # Store results
            results['optimized'][seq_len] = {
                'measurements': search_results['measurements'],
                'sla_results': search_results['sla_results'],
            }

            # Print summary
            print(f"\n  Summary for seq_len={seq_len}:")
            for sla_ms, sla_data in search_results['sla_results'].items():
                met_str = "✓" if sla_data['sla_met'] else "✗"
                print(f"    SLA {sla_ms}ms: bs={sla_data['max_batch_size']}, "
                      f"{sla_data['latency_ms']:.2f}ms (median: {sla_data['latency_median_ms']:.2f}ms), "
                      f"{sla_data['throughput']:.1f} seq/s {met_str}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results['optimized'][seq_len] = {
                'measurements': {},
                'sla_results': {},
            }
            torch.cuda.empty_cache()

    return results


def save_results(all_results: Dict, output_file: str):
    """Save aggregated results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results saved to {output_file}")


def save_raw_results_jsonl(all_results: Dict, output_file: str):
    """
    Save all raw data points to JSONL file.

    Each line contains one measurement with all metadata:
    {config, seq_len, batch_size, latency_ms, throughput, memory_mb}
    """
    jsonl_file = output_file.replace('.json', '_raw.jsonl')

    with open(jsonl_file, 'w') as f:
        for config_name, config_results in all_results.items():
            if config_name in ['20M', '120M', '720M']:  # Skip metadata fields
                estimated_params = config_results.get('estimated_params_m', 0)

                # Process optimized results
                for seq_len, seq_data in config_results.get('optimized', {}).items():
                    measurements = seq_data.get('measurements', {})

                    # Write all measurements
                    for batch_size, metrics in measurements.items():
                        record = {
                            'config': config_name,
                            'estimated_params_m': estimated_params,
                            'seq_len': int(seq_len),
                            'batch_size': int(batch_size),
                            'model_type': 'optimized',
                            'latency_ms': metrics['latency_ms'],
                            'latency_median_ms': metrics['latency_median_ms'],
                            'latency_std_ms': metrics['latency_std_ms'],
                            'throughput': metrics['throughput'],
                            'memory_mb': metrics['memory_mb'],
                        }
                        f.write(json.dumps(record) + '\n')

    print(f"Raw results saved to {jsonl_file}")


def main():
    parser = argparse.ArgumentParser(description='TabFormer Optimized Benchmark Sweep')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (default: {precision}_optimized_results.json)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--precision', type=str, choices=['bf16', 'fp32'], default='bf16',
                        help='Precision mode: bf16 or fp32 (default: bf16)')
    args = parser.parse_args()

    device = args.device
    use_bf16 = (args.precision == 'bf16')

    if args.output is None:
        args.output = f'{args.precision}_optimized_results.json'

    print(f"Running optimized benchmark sweep on {device}")
    print(f"Precision: {args.precision.upper()}")
    print("Using torch.compile(fullgraph=True, mode='reduce-overhead')")
    print("Binary search for batch sizes up to 128")
    print("Per-config recompilation for CUDA graph capture enabled")

    # Get model configs
    configs = get_model_configs()

    # SLA targets
    sla_targets_ms = [5.0, 10.0, 15.0]

    # Define sequence lengths per model (smaller models test longer sequences)
    model_seq_lengths = {
        '20M': [32, 64, 128, 256],
        '120M': [32, 64],
        '720M': [16],
    }

    print(f"\nSequence length configuration:")
    for model, seq_lens in model_seq_lengths.items():
        print(f"  {model}: {seq_lens}")
    print()

    # Run sweep for each model
    all_results = {}
    for config_name in ['20M', '120M', '720M']:
        config = configs[config_name]
        seq_lengths = model_seq_lengths[config_name]

        results = run_sweep(
            config_name,
            config,
            seq_lengths,
            sla_targets_ms,
            device=device,
            use_bf16=use_bf16,
        )
        all_results[config_name] = results

    # Save results
    save_results(all_results, args.output)
    save_raw_results_jsonl(all_results, args.output)

    print("\n" + "="*80)
    print("Optimized benchmark sweep complete!")
    print("="*80)


if __name__ == '__main__':
    main()
