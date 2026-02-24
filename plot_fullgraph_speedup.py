#!/usr/bin/env python3
"""
Plot speedup heatmap for SDPA+fullgraph results.
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

# Calculate speedups
speedup_data = defaultdict(lambda: defaultdict(dict))
for r in fullgraph_results:
    key = (r['model_type'], r['config_name'], r['batch_size'], r['seq_len'])
    if key in baseline_lookup:
        speedup = baseline_lookup[key]['avg_latency_ms'] / r['avg_latency_ms']
        model_key = f"{r['model_type']}_{r['config_name']}"
        speedup_data[model_key][r['batch_size']][r['seq_len']] = speedup

# Plot
models = ['bert_120M', 'bert_20M']
batch_sizes = sorted(set(r['batch_size'] for r in fullgraph_results))
seq_lens = sorted(set(r['seq_len'] for r in fullgraph_results))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('SDPA+torch.compile+fullgraph (reduce-overhead) Speedup vs Baseline',
             fontsize=16, fontweight='bold', y=0.98)

model_titles = {'bert_120M': 'BERT 120M', 'bert_20M': 'BERT 20M'}

for idx, model in enumerate(models):
    ax = axes[idx]

    # Create heatmap
    heatmap_data = np.zeros((len(batch_sizes), len(seq_lens)))
    for i, bs in enumerate(batch_sizes):
        for j, seq in enumerate(seq_lens):
            if bs in speedup_data[model] and seq in speedup_data[model][bs]:
                heatmap_data[i, j] = speedup_data[model][bs][seq]
            else:
                heatmap_data[i, j] = np.nan

    # Colormap centered at 1.0
    colors = ['#d73027', '#fc8d59', '#fee090', '#ffffff', '#e0f3f8', '#91bfdb', '#4575b4']
    cmap = LinearSegmentedColormap.from_list('speedup', colors, N=100)
    norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=10.0)

    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', norm=norm, interpolation='nearest')

    # Ticks
    ax.set_xticks(np.arange(len(seq_lens)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(seq_lens)
    ax.set_yticklabels(batch_sizes)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Batch Size', fontsize=11)

    # Title with avg speedup
    valid_speedups = heatmap_data[~np.isnan(heatmap_data)]
    avg_speedup = np.mean(valid_speedups) if len(valid_speedups) > 0 else 0
    ax.set_title(f'{model_titles[model]} (Avg: {avg_speedup:.2f}x)',
                 fontsize=12, fontweight='bold')

    # Annotations
    for i in range(len(batch_sizes)):
        for j in range(len(seq_lens)):
            if not np.isnan(heatmap_data[i, j]):
                speedup_val = heatmap_data[i, j]
                # Get baseline latency and optimized throughput
                model_type, config_name = model.split('_')
                key = (model_type, config_name, batch_sizes[i], seq_lens[j])
                baseline_lat = baseline_lookup[key]['avg_latency_ms'] if key in baseline_lookup else 0

                # Get optimized latency and throughput
                opt_result = next((r for r in fullgraph_results if
                                 r['model_type'] == model_type and
                                 r['config_name'] == config_name and
                                 r['batch_size'] == batch_sizes[i] and
                                 r['seq_len'] == seq_lens[j]), None)
                opt_lat = opt_result['avg_latency_ms'] if opt_result else 0
                opt_throughput = opt_result['throughput_samples_per_sec'] if opt_result else 0

                text_color = 'white' if speedup_val > 5.0 else 'black'
                ax.text(j, i, f'{speedup_val:.1f}x\n{baseline_lat:.1f}→{opt_lat:.1f}ms\n{opt_throughput:.0f}samp/s',
                       ha="center", va="center", color=text_color,
                       fontsize=8, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Speedup', fontsize=10)

plt.tight_layout()
plt.savefig('sdpa_fullgraph_speedup.png', dpi=300, bbox_inches='tight')
print("Saved: sdpa_fullgraph_speedup.png")
