# GPU Utilization Profiling Plan

## Context

Native PyTorch BERT with torch.compile(fullgraph=True, mode=reduce-overhead) achieves
2.6-2.7x speedup over HF BERT eager baseline. Next step: capture GPU utilization
time series to understand SM and memory efficiency.

## Clean Latency Results (seq_len=100, no nsys overhead)

| Config       | Variant                              | Avg (ms) | P50 (ms) | P99 (ms) | Throughput     | Speedup |
|--------------|--------------------------------------|----------|----------|----------|----------------|---------|
| 120M bs=32   | Optimized (native+compile+fullgraph) | 14.61    | 14.60    | 14.87    | 2,190 samples/s | 2.64x  |
| 120M bs=32   | Baseline (HF eager, no compile)      | 38.61    | 38.46    | 44.64    | 829 samples/s   | 1.0x   |
| 20M bs=36    | Optimized (native+compile+fullgraph) | 9.57     | 9.34     | 11.16    | 3,764 samples/s | 2.70x  |
| 20M bs=36    | Baseline (HF eager, no compile)      | 25.80    | 25.80    | 26.29    | 1,396 samples/s | 1.0x   |

## Prerequisites

- Pod relaunched WITHOUT dcgm hostengine auto-starting
- Start hostengine as root: `nv-hostengine`
- Verify: `dcgmi dmon -e 1002,1003,1004,1005 -c 3` should print SM metrics

## DCGM Fields to Collect

- 1002: SM Activity (% of time at least one warp is active)
- 1003: SM Occupancy (ratio of active warps to max warps)
- 1004: Tensor Core Activity
- 1005: DRAM Activity (memory bandwidth utilization)

## nvidia-smi dmon Fields

- `nvidia-smi dmon -s u -d 1` gives: sm%, mem%, enc%, dec% every 1 second

## 4 Runs to Profile

All runs: 200 iterations, 10 warmup iterations, seq_len=100.

### Run 1: 120M bs=32 Optimized
```bash
dcgmi dmon -e 1002,1003,1004,1005 -d 100 > dcgm_120M_optimized.csv &
nvidia-smi dmon -s u -d 1 -f smi_120M_optimized.csv &
python benchmark_sweep.py --model_types bert --config_names 120M --batch_sizes 32 --seq_lens 100 \
  --native_bert --torch_compile --torch_compile_mode reduce-overhead --torch_compile_fullgraph \
  --num_iterations 200 --warmup_iterations 10 --output_file bench_120M_optimized.json
kill %1 %2
```

### Run 2: 120M bs=32 Baseline
```bash
dcgmi dmon -e 1002,1003,1004,1005 -d 100 > dcgm_120M_baseline.csv &
nvidia-smi dmon -s u -d 1 -f smi_120M_baseline.csv &
python benchmark_sweep.py --model_types bert --config_names 120M --batch_sizes 32 --seq_lens 100 \
  --attn_impl eager --num_iterations 200 --warmup_iterations 10 --output_file bench_120M_baseline.json
kill %1 %2
```

### Run 3: 20M bs=36 Optimized
```bash
dcgmi dmon -e 1002,1003,1004,1005 -d 100 > dcgm_20M_optimized.csv &
nvidia-smi dmon -s u -d 1 -f smi_20M_optimized.csv &
python benchmark_sweep.py --model_types bert --config_names 20M --batch_sizes 36 --seq_lens 100 \
  --native_bert --torch_compile --torch_compile_mode reduce-overhead --torch_compile_fullgraph \
  --num_iterations 200 --warmup_iterations 10 --output_file bench_20M_optimized.json
kill %1 %2
```

### Run 4: 20M bs=36 Baseline
```bash
dcgmi dmon -e 1002,1003,1004,1005 -d 100 > dcgm_20M_baseline.csv &
nvidia-smi dmon -s u -d 1 -f smi_20M_baseline.csv &
python benchmark_sweep.py --model_types bert --config_names 20M --batch_sizes 36 --seq_lens 100 \
  --attn_impl eager --num_iterations 200 --warmup_iterations 10 --output_file bench_20M_baseline.json
kill %1 %2
```

## Post-Processing

1. Parse dcgm CSV files (skip header lines starting with #)
2. Parse nvidia-smi dmon CSV files
3. Plot time series:
   - SM Activity: optimized vs baseline (120M and 20M)
   - SM Occupancy: optimized vs baseline
   - Tensor Core Activity: optimized vs baseline
   - DRAM Activity: optimized vs baseline
   - GPU Memory Used over time
4. Generate summary statistics (mean, p50, p99 for each metric during steady state)
