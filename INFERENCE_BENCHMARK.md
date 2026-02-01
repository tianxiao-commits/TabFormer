# TabFormer Inference Benchmarking

This directory contains scripts for generating synthetic data and benchmarking TabFormer inference performance.

## Files

- `generate_synthetic_data.py` - Generate synthetic credit card transaction data
- `benchmark_inference.py` - Benchmark inference throughput and latency
- `test-pod.yaml` - Kubernetes pod configuration with automated benchmarking

## Quick Start

### 1. Generate Synthetic Dataset

```bash
python generate_synthetic_data.py --nrows 50000 --nusers 500
```

Options:
- `--nrows`: Number of transaction rows to generate (default: 10000)
- `--nusers`: Number of unique users (default: 100)
- `--output`: Output CSV file path (default: ./data/credit_card/card_transaction.v1.csv)
- `--seed`: Random seed for reproducibility (default: 42)

### 2. Run Inference Benchmark

```bash
python benchmark_inference.py \
  --lm_type bert \
  --field_ce \
  --field_hs 64 \
  --data_root ./data/credit_card/ \
  --nrows 10000 \
  --max_samples 1000 \
  --batch_size 32
```

Options:
- `--lm_type`: Model type (bert or gpt2)
- `--batch_size`: Batch size for inference (default: 32)
- `--max_samples`: Maximum samples to benchmark (default: 1000)
- `--nrows`: Number of rows to load from dataset (default: 10000)
- `--model_dir`: Path to model checkpoint directory (default: ./checkpoints)
- `--output_file`: JSON file for results (default: benchmark_results.json)

### 3. Deploy on Kubernetes

The pod will automatically:
1. Install dependencies
2. Clone the repository
3. Generate synthetic dataset (50K transactions)
4. Run inference benchmark
5. Save results to `/workspace/benchmark_results.json`

```bash
# Delete old pod if exists
kubectl delete pod tabformer-a100-pod -n rlvr

# Deploy new pod
kubectl apply -f test-pod.yaml

# Monitor logs
kubectl logs -f tabformer-a100-pod -n rlvr

# Check results
kubectl exec tabformer-a100-pod -n rlvr -- cat /workspace/benchmark_results.json
```

## Benchmark Metrics

The benchmark reports:

**Latency (per batch)**:
- Average latency
- P50, P95, P99 percentiles

**Throughput**:
- Samples per second
- Total processing time

**GPU Memory**:
- Allocated memory
- Reserved memory
- Peak memory usage

## Example Output

```json
{
  "timestamp": "2026-02-01T17:30:00",
  "model": "bert",
  "device": "cuda:0",
  "batch_size": 32,
  "total_samples": 1000,
  "avg_latency_ms": 45.23,
  "p50_latency_ms": 44.12,
  "p95_latency_ms": 52.34,
  "p99_latency_ms": 58.91,
  "avg_throughput_samples_per_sec": 707.82
}
```

## Training After Benchmark

After benchmarking, you can train a model:

```bash
cd /workspace/TabFormer
python main.py \
  --do_train \
  --mlm \
  --field_ce \
  --lm_type bert \
  --field_hs 64 \
  --data_type card \
  --nrows 50000 \
  --num_train_epochs 3 \
  --output_dir ./checkpoints/model
```

## Troubleshooting

### Pod Errors
- Check logs: `kubectl logs tabformer-a100-pod -n rlvr`
- Describe pod: `kubectl describe pod tabformer-a100-pod -n rlvr`

### Dataset Issues
- Ensure synthetic data is generated before running benchmark
- Check data format matches expected schema

### GPU Issues
- Verify GPU availability: `nvidia-smi`
- Check CUDA version compatibility with PyTorch
