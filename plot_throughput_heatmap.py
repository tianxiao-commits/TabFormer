#!/usr/bin/env python3
"""
Plot throughput and speedup heatmaps for SDPA+fullgraph results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from collections import defaultdict

# Load baseline results
with open('benchmark_sweep_results.json', 'r') as f:
    baseline_results = json.load(f)

# Load fullgraph results
with open('benchmark_sweep_fullgraph_sdpa.json', 'r') as f:
    results = json.load(f)

# Create baseline lookup
baseline_lookup = {}
for result in baseline_results:
    key = (result['model_type'], result['config_name'],
           result['batch_size'], result['seq_len'])
    baseline_lookup[key] = result

# Organize throughput and speedup data
throughput_data = defaultdict(lambda: defaultdict(dict))
speedup_data = defaultdict(lambda: defaultdict(dict))

for result in results:
    model_key = f"{result['model_type']}_{result['config_name']}"
    key = (result['model_type'], result['config_name'], result['batch_size'], result['seq_len'])

    throughput_data[model_key][result['batch_size']][result['seq_len']] = result['throughput_samples_per_sec']

    if key in baseline_lookup:
        speedup = baseline_lookup[key]['avg_latency_ms'] / result['avg_latency_ms']
        speedup_data[model_key][result['batch_size']][result['seq_len']] = speedup

# Prepare for plotting
models = ['bert_120M', 'bert_20M']
batch_sizes = sorted(set(r['batch_size'] for r in results))
seq_lens = sorted(set(r['seq_len'] for r in results))

# Create speedup heatmap
fig_speedup, axes_speedup = plt.subplots(1, 2, figsize=(14, 6))
fig_speedup.suptitle('SDPA+torch.compile+fullgraph (reduce-overhead) Speedup vs Baseline',
                     fontsize=16, fontweight='bold', y=0.98)

# Create throughput heatmap
fig_through, axes_through = plt.subplots(1, 2, figsize=(14, 6))
fig_through.suptitle('SDPA+torch.compile+fullgraph (reduce-overhead) Throughput (samples/sec)',
                     fontsize=16, fontweight='bold', y=0.98)

model_titles = {
    'bert_120M': 'BERT 120M',
    'bert_20M': 'BERT 20M'
}

for idx, model in enumerate(models):
    ax_speedup = axes_speedup[idx]
    ax_through = axes_through[idx]

    # Create heatmap matrices
    speedup_heatmap = np.zeros((len(batch_sizes), len(seq_lens)))
    throughput_heatmap = np.zeros((len(batch_sizes), len(seq_lens)))

    for i, bs in enumerate(batch_sizes):
        for j, seq in enumerate(seq_lens):
            if bs in throughput_data[model] and seq in throughput_data[model][bs]:
                heatmap_data[i, j] = throughput_data[model][bs][seq]
            else:
                heatmap_data[i, j] = np.nan

    # Calculate average throughput
    valid_throughputs = heatmap_data[~np.isnan(heatmap_data)]
    avg_throughput = np.mean(valid_throughputs) if len(valid_throughputs) > 0 else 0

    # Use sequential colormap for throughput (higher is better)
    cmap = plt.cm.YlGnBu

    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(seq_lens)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(seq_lens)
    ax.set_yticklabels(batch_sizes)

    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Batch Size', fontsize=11)
    ax.set_title(f'{model_titles[model]} (Avg: {avg_throughput:.0f} samples/sec)',
                 fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(seq_lens)):
            if not np.isnan(heatmap_data[i, j]):
                throughput_val = heatmap_data[i, j]
                # Use white text for dark cells, black for light cells
                text_color = 'white' if throughput_val < np.max(valid_throughputs) * 0.4 else 'black'
                ax.text(j, i, f'{throughput_val:.0f}',
                       ha="center", va="center", color=text_color,
                       fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Throughput (samples/sec)', fontsize=10)

plt.tight_layout()
plt.savefig('sdpa_fullgraph_throughput.png', dpi=300, bbox_inches='tight')
print("Saved throughput heatmap to sdpa_fullgraph_throughput.png")

# Print summary
print("\nThroughput Summary:")
for model in models:
    throughputs = []
    for bs in throughput_data[model]:
        for seq in throughput_data[model][bs]:
            throughputs.append(throughput_data[model][bs][seq])

    if throughputs:
        print(f"\n{model_titles[model]}:")
        print(f"  Average: {np.mean(throughputs):.0f} samples/sec")
        print(f"  Min: {np.min(throughputs):.0f} samples/sec")
        print(f"  Max: {np.max(throughputs):.0f} samples/sec")
