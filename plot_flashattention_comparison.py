#!/usr/bin/env python3
"""
Plot FlashAttention-2 speedup compared to baseline vanilla attention.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from collections import defaultdict

# Load baseline results
with open('benchmark_sweep_results.json', 'r') as f:
    baseline_results = json.load(f)

# Load FlashAttention BERT results
with open('benchmark_sweep_results_bert_flash.json', 'r') as f:
    bert_flash_results = json.load(f)

# Load FlashAttention GPT2 results
with open('benchmark_sweep_results_flash.json', 'r') as f:
    gpt2_flash_results = json.load(f)

# Combine FlashAttention results
flash_results = bert_flash_results + gpt2_flash_results

# Create lookup for baseline results
baseline_lookup = {}
for result in baseline_results:
    key = (result['model_type'], result['config_name'],
           result['batch_size'], result['seq_len'])
    baseline_lookup[key] = result

# Calculate speedups
speedup_data = defaultdict(lambda: defaultdict(dict))

for result in flash_results:
    key = (result['model_type'], result['config_name'],
           result['batch_size'], result['seq_len'])

    if key in baseline_lookup:
        baseline = baseline_lookup[key]
        speedup = baseline['avg_latency_ms'] / result['avg_latency_ms']

        model_key = f"{result['model_type']}_{result['config_name']}"
        speedup_data[model_key][result['batch_size']][result['seq_len']] = speedup

# Prepare data for plotting
models = ['bert_120M', 'bert_20M', 'gpt2_120M', 'gpt2_20M']
batch_sizes = sorted(set(r['batch_size'] for r in flash_results))
seq_lens = sorted(set(r['seq_len'] for r in flash_results))

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('FlashAttention-2 Speedup vs Baseline (Vanilla Attention)',
             fontsize=16, fontweight='bold', y=0.995)

model_titles = {
    'bert_120M': 'BERT 120M',
    'bert_20M': 'BERT 20M',
    'gpt2_120M': 'GPT2 120M',
    'gpt2_20M': 'GPT2 20M'
}

for idx, model in enumerate(models):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # Create heatmap matrix
    heatmap_data = np.zeros((len(batch_sizes), len(seq_lens)))

    for i, bs in enumerate(batch_sizes):
        for j, seq in enumerate(seq_lens):
            if bs in speedup_data[model] and seq in speedup_data[model][bs]:
                heatmap_data[i, j] = speedup_data[model][bs][seq]
            else:
                heatmap_data[i, j] = np.nan

    # Calculate average speedup (excluding NaN)
    valid_speedups = heatmap_data[~np.isnan(heatmap_data)]
    avg_speedup = np.mean(valid_speedups) if len(valid_speedups) > 0 else 0

    # Create custom diverging colormap centered at 1.0
    # Define colors: dark red (slow) -> white (1.0x) -> dark blue (fast)
    colors = ['#d73027', '#fc8d59', '#fee090', '#ffffff', '#e0f3f8', '#91bfdb', '#4575b4']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('speedup', colors, N=n_bins)

    # Use TwoSlopeNorm to center white at 1.0
    vmin = 0.7
    vmax = 3.5
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto',
                   norm=norm, interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(seq_lens)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(seq_lens)
    ax.set_yticklabels(batch_sizes)

    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Batch Size', fontsize=11)

    # Calculate average baseline latency
    baseline_latencies = []
    for i, bs in enumerate(batch_sizes):
        for j, seq in enumerate(seq_lens):
            model_type, config_name = model.split('_')
            key = (model_type, config_name, bs, seq)
            if key in baseline_lookup:
                baseline_latencies.append(baseline_lookup[key]['avg_latency_ms'])
    avg_baseline = np.mean(baseline_latencies) if baseline_latencies else 0

    ax.set_title(f'{model_titles[model]} (Avg: {avg_speedup:.2f}x, Baseline: {avg_baseline:.1f}ms)',
                 fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(seq_lens)):
            if not np.isnan(heatmap_data[i, j]):
                speedup_val = heatmap_data[i, j]

                # Get baseline latency for this cell
                model_type, config_name = model.split('_')
                bs = batch_sizes[i]
                seq = seq_lens[j]
                key = (model_type, config_name, bs, seq)
                baseline_lat = baseline_lookup[key]['avg_latency_ms'] if key in baseline_lookup else 0

                # Use black text for values near 1.0, white for extremes
                if 0.85 <= speedup_val <= 1.15:
                    text_color = 'black'
                elif speedup_val < 0.80 or speedup_val > 2.0:
                    text_color = 'white'
                else:
                    text_color = 'black'

                # Show speedup and baseline latency
                ax.text(j, i, f'{speedup_val:.2f}x\n({baseline_lat:.1f}ms)',
                       ha="center", va="center", color=text_color,
                       fontsize=8, fontweight='bold')

    # Add colorbar for each subplot
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Speedup', fontsize=10)

    # Add horizontal line at 1.0x on colorbar
    cbar.ax.axhline(y=1.0, color='black', linewidth=2, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('flashattention_speedup.png', dpi=300, bbox_inches='tight')
print("Saved speedup heatmap to flashattention_speedup.png")

# Print summary statistics
print("\n" + "="*70)
print("FlashAttention-2 Performance Summary")
print("="*70)

for model in models:
    print(f"\n{model_titles[model]}:")
    valid_speedups = []
    for bs in speedup_data[model]:
        for seq in speedup_data[model][bs]:
            valid_speedups.append(speedup_data[model][bs][seq])

    if valid_speedups:
        print(f"  Average speedup: {np.mean(valid_speedups):.2f}x")
        print(f"  Min speedup: {np.min(valid_speedups):.2f}x")
        print(f"  Max speedup: {np.max(valid_speedups):.2f}x")
        print(f"  Median speedup: {np.median(valid_speedups):.2f}x")

        # Count faster vs slower
        faster = sum(1 for s in valid_speedups if s > 1.0)
        slower = sum(1 for s in valid_speedups if s < 1.0)
        print(f"  Faster configs: {faster}/{len(valid_speedups)}")
        print(f"  Slower configs: {slower}/{len(valid_speedups)}")

print("\n" + "="*70)
