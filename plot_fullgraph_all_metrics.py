#!/usr/bin/env python3
"""
Plot 4 heatmaps: speedup, baseline latency, optimized latency, and throughput.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from collections import defaultdict

# Load baseline
with open('benchmark_sweep_results.json', 'r') as f:
    baseline_results = json.load(f)

# Load fullgraph
with open('benchmark_sweep_fullgraph_sdpa.json', 'r') as f:
    fullgraph_results = json.load(f)

# Create lookups
baseline_lookup = {}
for r in baseline_results:
    key = (r['model_type'], r['config_name'], r['batch_size'], r['seq_len'])
    baseline_lookup[key] = r

# Organize all data
speedup_data = defaultdict(lambda: defaultdict(dict))
baseline_lat_data = defaultdict(lambda: defaultdict(dict))
optimized_lat_data = defaultdict(lambda: defaultdict(dict))
baseline_throughput_data = defaultdict(lambda: defaultdict(dict))
optimized_throughput_data = defaultdict(lambda: defaultdict(dict))

for r in fullgraph_results:
    key = (r['model_type'], r['config_name'], r['batch_size'], r['seq_len'])
    model_key = f"{r['model_type']}_{r['config_name']}"

    if key in baseline_lookup:
        baseline_lat = baseline_lookup[key]['avg_latency_ms']
        baseline_throughput = baseline_lookup[key]['throughput_samples_per_sec']
        opt_lat = r['avg_latency_ms']
        opt_throughput = r['throughput_samples_per_sec']
        speedup = baseline_lat / opt_lat

        speedup_data[model_key][r['batch_size']][r['seq_len']] = speedup
        baseline_lat_data[model_key][r['batch_size']][r['seq_len']] = baseline_lat
        optimized_lat_data[model_key][r['batch_size']][r['seq_len']] = opt_lat
        baseline_throughput_data[model_key][r['batch_size']][r['seq_len']] = baseline_throughput
        optimized_throughput_data[model_key][r['batch_size']][r['seq_len']] = opt_throughput

# Plot setup
models = ['bert_120M', 'bert_20M']
batch_sizes = sorted(set(r['batch_size'] for r in fullgraph_results))
seq_lens = sorted(set(r['seq_len'] for r in fullgraph_results))
model_titles = {'bert_120M': 'BERT 120M', 'bert_20M': 'BERT 20M'}

# Create 2x5 grid (2 models x 5 metrics)
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle('SDPA+torch.compile+fullgraph (reduce-overhead) Performance Metrics',
             fontsize=16, fontweight='bold', y=0.995)

metric_titles = ['Speedup vs Baseline', 'Baseline Latency (ms)',
                 'Optimized Latency (ms)', 'Baseline Throughput (samp/s)', 'Optimized Throughput (samp/s)']
data_sources = [speedup_data, baseline_lat_data, optimized_lat_data, baseline_throughput_data, optimized_throughput_data]

for model_idx, model in enumerate(models):
    for metric_idx, (data_source, metric_title) in enumerate(zip(data_sources, metric_titles)):
        ax = axes[model_idx, metric_idx]

        # Create heatmap
        heatmap_data = np.zeros((len(batch_sizes), len(seq_lens)))
        for i, bs in enumerate(batch_sizes):
            for j, seq in enumerate(seq_lens):
                if bs in data_source[model] and seq in data_source[model][bs]:
                    heatmap_data[i, j] = data_source[model][bs][seq]
                else:
                    heatmap_data[i, j] = np.nan

        # Choose colormap and normalization based on metric
        if metric_idx == 0:  # Speedup - use light pastel colors
            colors = ['#ffe0e0', '#fff0f0', '#ffffff', '#f0f8ff', '#e0f0ff', '#d0e8ff']
            cmap = LinearSegmentedColormap.from_list('speedup', colors, N=100)
            norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=10.0)
        else:  # Latency or throughput - use very light single color gradients
            if metric_idx in [3, 4]:  # Throughput (both baseline and optimized)
                colors = ['#ffffff', '#e8f4f8', '#d0e9f0', '#b8dee8']
            else:  # Latency
                colors = ['#ffffff', '#fff5e0', '#ffebc0', '#ffe0a0']
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
            norm = None

        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', norm=norm, interpolation='nearest')

        # Ticks
        ax.set_xticks(np.arange(len(seq_lens)))
        ax.set_yticks(np.arange(len(batch_sizes)))
        ax.set_xticklabels(seq_lens)
        ax.set_yticklabels(batch_sizes)

        if model_idx == 1:  # Bottom row
            ax.set_xlabel('Sequence Length', fontsize=10)
        if metric_idx == 0:  # Left column
            ax.set_ylabel('Batch Size', fontsize=10)

        # Title
        valid_data = heatmap_data[~np.isnan(heatmap_data)]
        avg_val = np.mean(valid_data) if len(valid_data) > 0 else 0

        if metric_idx == 0:
            title = f'{model_titles[model]}: {metric_title}\n(Avg: {avg_val:.2f}x)'
        elif metric_idx in [1, 2]:
            title = f'{metric_title}\n(Avg: {avg_val:.2f}ms)'
        else:  # Throughput metrics
            title = f'{metric_title}\n(Avg: {avg_val:.0f})'

        ax.set_title(title, fontsize=10, fontweight='bold')

        # Annotations
        for i in range(len(batch_sizes)):
            for j in range(len(seq_lens)):
                if not np.isnan(heatmap_data[i, j]):
                    val = heatmap_data[i, j]

                    # Text color - always use black for light backgrounds
                    text_color = 'black'
                    if metric_idx == 0:  # Speedup
                        text = f'{val:.1f}x'
                    elif metric_idx in [1, 2]:  # Latency
                        text = f'{val:.1f}'
                    else:  # Throughput (baseline and optimized)
                        text = f'{val:.0f}'

                    ax.text(j, i, text, ha="center", va="center",
                           color=text_color, fontsize=8, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('sdpa_fullgraph_all_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: sdpa_fullgraph_all_metrics.png")
