#!/usr/bin/env python3
"""
Compare SDPA+compile+reduce_overhead vs SDPA+compile regular mode.
"""

import json
import numpy as np

# Load baseline results
with open('benchmark_sweep_results.json', 'r') as f:
    baseline_results = json.load(f)

# Load regular SDPA+compile results
with open('benchmark_sweep_results_sdpa_compile.json', 'r') as f:
    sdpa_regular = json.load(f)

# Load reduce-overhead results
with open('benchmark_sweep_results_sdpa_compile_reduce_overhead.json', 'r') as f:
    sdpa_reduce = json.load(f)

# Create lookups
baseline_lookup = {}
for result in baseline_results:
    key = (result['model_type'], result['config_name'],
           result['batch_size'], result['seq_len'])
    baseline_lookup[key] = result

sdpa_regular_lookup = {}
for result in sdpa_regular:
    key = (result['model_type'], result['config_name'],
           result['batch_size'], result['seq_len'])
    sdpa_regular_lookup[key] = result

# Compare results
print("="*90)
print("SDPA+compile: Regular vs Reduce-Overhead Mode Comparison")
print("="*90)
print(f"{'Model':<12} {'BS':>3} {'Seq':>4} | {'Baseline':>9} | {'Regular':>9} {'Speedup':>8} | {'ReduceOH':>9} {'Speedup':>8} | {'Improve':>8}")
print("-"*90)

for result in sdpa_reduce:
    key = (result['model_type'], result['config_name'],
           result['batch_size'], result['seq_len'])

    if key in baseline_lookup and key in sdpa_regular_lookup:
        baseline = baseline_lookup[key]
        regular = sdpa_regular_lookup[key]

        model_name = f"{result['model_type']}-{result['config_name']}"
        bs = result['batch_size']
        seq = result['seq_len']

        baseline_lat = baseline['avg_latency_ms']
        regular_lat = regular['avg_latency_ms']
        reduce_lat = result['avg_latency_ms']

        regular_speedup = baseline_lat / regular_lat
        reduce_speedup = baseline_lat / reduce_lat
        improvement = reduce_lat / regular_lat  # <1.0 means reduce-overhead is faster

        print(f"{model_name:<12} {bs:>3} {seq:>4} | {baseline_lat:>8.2f}ms | "
              f"{regular_lat:>8.2f}ms {regular_speedup:>7.2f}x | "
              f"{reduce_lat:>8.2f}ms {reduce_speedup:>7.2f}x | "
              f"{improvement:>7.2f}x")

# Summary statistics
print("\n" + "="*90)
print("Summary by Model")
print("="*90)

from collections import defaultdict
stats = defaultdict(lambda: {'regular_speedups': [], 'reduce_speedups': [], 'improvements': []})

for result in sdpa_reduce:
    key = (result['model_type'], result['config_name'],
           result['batch_size'], result['seq_len'])

    if key in baseline_lookup and key in sdpa_regular_lookup:
        baseline = baseline_lookup[key]
        regular = sdpa_regular_lookup[key]

        model_key = f"{result['model_type']}_{result['config_name']}"

        regular_speedup = baseline['avg_latency_ms'] / regular['avg_latency_ms']
        reduce_speedup = baseline['avg_latency_ms'] / result['avg_latency_ms']
        improvement = result['avg_latency_ms'] / regular['avg_latency_ms']

        stats[model_key]['regular_speedups'].append(regular_speedup)
        stats[model_key]['reduce_speedups'].append(reduce_speedup)
        stats[model_key]['improvements'].append(improvement)

for model_key in sorted(stats.keys()):
    print(f"\n{model_key}:")
    reg_speedups = stats[model_key]['regular_speedups']
    red_speedups = stats[model_key]['reduce_speedups']
    improvements = stats[model_key]['improvements']

    print(f"  Regular mode:        Avg {np.mean(reg_speedups):.2f}x  (range: {np.min(reg_speedups):.2f}x - {np.max(reg_speedups):.2f}x)")
    print(f"  Reduce-overhead:     Avg {np.mean(red_speedups):.2f}x  (range: {np.min(red_speedups):.2f}x - {np.max(red_speedups):.2f}x)")
    print(f"  Improvement:         Avg {np.mean(improvements):.2f}x  (range: {np.min(improvements):.2f}x - {np.max(improvements):.2f}x)")

    better = sum(1 for i in improvements if i < 1.0)
    worse = sum(1 for i in improvements if i > 1.0)
    print(f"  Reduce-overhead better in {better}/{len(improvements)} configs, worse in {worse}/{len(improvements)}")

print("\n" + "="*90)
