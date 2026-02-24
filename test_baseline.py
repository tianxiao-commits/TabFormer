"""
Test baseline (no torch.compile) for nsys profiling.
Config: 20M model, seq_len=128, batch_size=8, BF16
"""

import torch
import torch.nn as nn
import time
from benchmark_tabformer_sweep import TabFormerNative, get_model_configs, create_sample_input

def main():
    device = 'cuda'
    batch_size = 8
    seq_len = 128
    warmup_iters = 20
    bench_iters = 50
    use_bf16 = True

    print("="*80)
    print("BASELINE (no torch.compile)")
    print("="*80)
    print(f"Config: 20M model")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Precision: BF16")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")
    print("="*80)

    configs = get_model_configs()
    config = configs['20M']

    # Create baseline model
    model = TabFormerNative(config).to(device)
    model.eval()

    if use_bf16:
        model = model.bfloat16()

    input_ids = create_sample_input(batch_size, seq_len, device)

    # Warmup
    print(f"Warming up...")
    with torch.no_grad():
        for i in range(warmup_iters):
            _ = model(input_ids)
            if i == 0:
                print(f"  First iteration complete")

    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking...")
    latencies = []
    with torch.no_grad():
        for _ in range(bench_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    mean_latency = sum(latencies) / len(latencies)
    throughput = batch_size * 1000 / mean_latency
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("="*80)
    print(f"Mean latency:    {mean_latency:.3f}ms")
    print(f"Throughput:      {throughput:.1f} seq/s")
    print(f"Peak memory:     {memory_mb:.1f}MB")
    print("="*80)


if __name__ == '__main__':
    main()
