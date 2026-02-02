#!/usr/bin/env bash
# Best benchmark configuration: native BERT + SDPA + torch.compile(fullgraph=True, reduce-overhead)

python benchmark_sweep.py \
  --model_types bert \
  --config_names 120M,20M \
  --batch_sizes 1,4,16 \
  --seq_lens 10,20,40,80 \
  --native_bert \
  --torch_compile \
  --torch_compile_mode reduce-overhead \
  --torch_compile_fullgraph \
  --num_iterations 100 \
  --warmup_iterations 10 \
  --output_file benchmark_sweep_fullgraph_sdpa.json
