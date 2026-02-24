# TabFormer Benchmark Instructions

## Overview

This document explains how to run the comprehensive TabFormer benchmark sweep on A100 GPU.

## What the Benchmark Does

### Configurations Tested
- **20M TabFormer**: 4 field layers, 8 sequence layers
- **120M TabFormer**: 6 field layers, 10 sequence layers
- **720M TabFormer**: 6 field layers, 18 sequence layers

### Sweep Parameters
- **Sequence Lengths**: 32, 64, 128, 256, 512, 1024
- **Batch Sizes**: 1, 2, 4, 8, 16, 32
- **Iterations**: 100 warmup + 100 benchmark per config

### Comparison
- **Baseline**: Native PyTorch model (no torch.compile)
- **Optimized**: torch.compile(fullgraph=True, mode='reduce-overhead')

### SLA Analysis
Computes maximum throughput achievable for:
- 5ms latency target
- 10ms latency target
- 15ms latency target

### Input Data Format
Each transaction has 14 fields:
- 2 categorical fields (300 categories each)
- 2 categorical fields (30 categories each)
- 2 numerical fields (200 buckets each)
- 8 numerical fields (20 buckets each)

## Launch Benchmark on A100

### 1. Deploy the Pod

```bash
kubectl apply -f benchmark-pod-a100.yaml
```

### 2. Monitor Progress

```bash
# Watch pod status
kubectl get pods -n rlvr -w

# View logs
kubectl logs -n rlvr tabformer-benchmark-a100 -f
```

### 3. Retrieve Results

Once the benchmark completes (pod shows "Running" but benchmark is done):

```bash
# Copy results to local machine
kubectl cp rlvr/tabformer-benchmark-a100:/workspace/tabformer_sweep_results_a100.json ./tabformer_sweep_results_a100.json
```

### 4. Clean Up

```bash
# Delete the pod when done
kubectl delete pod -n rlvr tabformer-benchmark-a100
```

## Expected Runtime

- **20M config**: ~30-60 minutes
- **120M config**: ~1-2 hours
- **720M config**: ~3-5 hours
- **Total**: ~5-8 hours for all three configs

## Results Format

The output JSON file contains:

```json
{
  "20M": {
    "config_name": "20M",
    "estimated_params_m": 20.5,
    "baseline": {
      "32": {
        "1": {
          "latency_ms": 2.5,
          "throughput": 400.0,
          "memory_mb": 1024
        },
        ...
      },
      ...
    },
    "optimized": {
      ...
    },
    "sla_throughput": {
      "baseline": {
        "5.0": {
          "32": {
            "max_throughput": 400.0,
            "best_batch_size": 1
          },
          ...
        },
        ...
      },
      "optimized": {
        ...
      }
    }
  },
  "120M": { ... },
  "720M": { ... }
}
```

## Manual Run (if pod fails)

If you need to run manually in an existing pod:

```bash
# Connect to pod
kubectl exec -it -n rlvr tabformer-benchmark-a100 -- /bin/bash

# Navigate to TabFormer
cd /workspace/TabFormer

# Run benchmark
python benchmark_tabformer_sweep.py \
  --configs 20M 120M 720M \
  --seq-lengths 32 64 128 256 512 1024 \
  --batch-sizes 1 2 4 8 16 32 \
  --sla-targets 5.0 10.0 15.0 \
  --output /workspace/tabformer_sweep_results_a100.json
```

## Analyzing Results

After downloading the results, you can analyze them with:

```python
import json
import pandas as pd

# Load results
with open('tabformer_sweep_results_a100.json') as f:
    results = json.load(f)

# Extract SLA throughput for 10ms target
for config_name in ['20M', '120M', '720M']:
    print(f"\n{config_name} - 10ms SLA Throughput:")
    sla_data = results[config_name]['sla_throughput']

    for seq_len in [32, 64, 128, 256, 512, 1024]:
        baseline = sla_data['baseline'][10.0][str(seq_len)]
        optimized = sla_data['optimized'][10.0][str(seq_len)]

        speedup = optimized['max_throughput'] / baseline['max_throughput'] if baseline['max_throughput'] > 0 else 0

        print(f"  Seq {seq_len}: Baseline {baseline['max_throughput']:.1f} seq/s "
              f"→ Optimized {optimized['max_throughput']:.1f} seq/s ({speedup:.2f}x)")
```

## Troubleshooting

### Out of Memory (OOM)

If you hit OOM errors:
- Reduce batch sizes: `--batch-sizes 1 2 4 8`
- Test fewer sequence lengths: `--seq-lengths 32 64 128 256`
- Test one config at a time: `--configs 20M`

### Benchmark Too Slow

To speed up testing:
- Reduce iterations: Modify script to use 50 warmup + 50 benchmark
- Test fewer configs: `--configs 20M 120M`

### Pod Not Starting

Check GPU availability:
```bash
kubectl get nodes -o json | jq '.items[] | select(.status.allocatable."nvidia.com/gpu" != null) | {name:.metadata.name, gpu:.status.allocatable."nvidia.com/gpu"}'
```

## Architecture Details

### Baseline (No Compile)
```
Input → Field Embeddings → Field Transformer (Bidirectional)
  → Concatenate & Project → Sequence Transformer (Causal GPT with RoPE)
  → Output Projection
```

All using:
- RMSNorm (FP32 computation)
- RoPE position embeddings (FP32 cos/sin tables)
- Linear/embeddings in BF16 (with autocast)

### Optimized (With Compile)
Same architecture but with:
- `torch.compile(fullgraph=True, mode='reduce-overhead')`
- CUDA graph capture during warmup
- Kernel fusion optimizations

## Next Steps

After getting results:
1. Analyze speedups across different configs
2. Identify optimal batch sizes for each SLA target
3. Compare against HuggingFace baseline (if available)
4. Profile with nsys for detailed kernel analysis
