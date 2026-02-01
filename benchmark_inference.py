"""
Inference benchmarking script for TabFormer models.
Measures throughput, latency, and GPU utilization during inference.
"""
import os
from os.path import join
import logging
import numpy as np
import torch
import time
import argparse
from datetime import datetime

from transformers import DataCollatorForLanguageModeling

from dataset.card import TransactionDataset
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import TransDataCollatorForLanguageModeling

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class InferenceBenchmark:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset = None
        self.vocab = None

    def load_model(self):
        """Load the trained model or initialize a new one."""
        logger.info(f"Loading model from {self.args.model_dir}")

        # Load dataset to get vocab
        dataset = TransactionDataset(
            root=self.args.data_root,
            fname=self.args.data_fname,
            fextension=self.args.data_extension,
            vocab_dir=self.args.model_dir,
            nrows=self.args.nrows,
            mlm=(self.args.lm_type == 'bert'),
            cached=self.args.cached,
            stride=self.args.stride,
            flatten=self.args.flatten,
            return_labels=False,
            skip_user=self.args.skip_user
        )

        self.vocab = dataset.vocab
        custom_special_tokens = self.vocab.get_special_tokens()

        # Initialize model
        if self.args.lm_type == "bert":
            tab_net = TabFormerBertLM(
                custom_special_tokens,
                vocab=self.vocab,
                field_ce=self.args.field_ce,
                flatten=self.args.flatten,
                ncols=dataset.ncols,
                field_hidden_size=self.args.field_hs
            )
        else:
            tab_net = TabFormerGPT2(
                custom_special_tokens,
                vocab=self.vocab,
                field_ce=self.args.field_ce,
                flatten=self.args.flatten,
            )

        # Try to load checkpoint if exists
        checkpoint_path = join(self.args.model_dir, 'pytorch_model.bin')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            tab_net.model.load_state_dict(state_dict)
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, using randomly initialized model")

        self.model = tab_net.model.to(self.device)
        self.model.eval()

        # Use small subset for inference
        total_len = len(dataset)
        test_len = min(self.args.max_samples, total_len)
        self.dataset = torch.utils.data.Subset(dataset, range(test_len))

        logger.info(f"Model loaded: {self.model.__class__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dataset size: {len(self.dataset)}")

        return tab_net.tokenizer

    def warmup(self, tokenizer, num_warmup=10):
        """Warmup GPU before benchmarking."""
        logger.info(f"Warming up with {num_warmup} iterations...")

        if self.args.flatten:
            collator_cls = DataCollatorForLanguageModeling
        else:
            collator_cls = TransDataCollatorForLanguageModeling

        data_collator = collator_cls(
            tokenizer=tokenizer,
            mlm=(self.args.lm_type == 'bert'),
            mlm_probability=0.15
        )

        with torch.no_grad():
            for i in range(min(num_warmup, len(self.dataset))):
                sample = self.dataset[i]
                if isinstance(sample, tuple):
                    sample = sample[0]

                batch = data_collator([sample])
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.model(**batch)
                else:
                    batch = batch.to(self.device)
                    self.model(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info("Warmup complete")

    def benchmark(self, tokenizer):
        """Run inference benchmark."""
        logger.info(f"\n{'=' * 60}")
        logger.info("Starting inference benchmark")
        logger.info(f"{'=' * 60}")

        if self.args.flatten:
            collator_cls = DataCollatorForLanguageModeling
        else:
            collator_cls = TransDataCollatorForLanguageModeling

        data_collator = collator_cls(
            tokenizer=tokenizer,
            mlm=(self.args.lm_type == 'bert'),
            mlm_probability=0.15
        )

        # Benchmark metrics
        latencies = []
        throughputs = []

        total_samples = len(self.dataset)
        logger.info(f"Benchmarking {total_samples} samples with batch size {self.args.batch_size}")

        with torch.no_grad():
            for batch_start in range(0, total_samples, self.args.batch_size):
                batch_end = min(batch_start + self.args.batch_size, total_samples)
                batch_samples = []

                for i in range(batch_start, batch_end):
                    sample = self.dataset[i]
                    if isinstance(sample, tuple):
                        sample = sample[0]
                    batch_samples.append(sample)

                # Prepare batch
                batch = data_collator(batch_samples)
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)

                # Measure inference time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.time()

                if isinstance(batch, dict):
                    _ = self.model(**batch)
                else:
                    _ = self.model(batch)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()

                # Calculate metrics
                latency = (end_time - start_time) * 1000  # ms
                actual_batch_size = len(batch_samples)
                throughput = actual_batch_size / (end_time - start_time)  # samples/sec

                latencies.append(latency)
                throughputs.append(throughput)

                if (batch_start // self.args.batch_size) % 10 == 0:
                    logger.info(f"Processed {batch_end}/{total_samples} samples | "
                                f"Latency: {latency:.2f}ms | "
                                f"Throughput: {throughput:.2f} samples/s")

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        avg_throughput = np.mean(throughputs)

        logger.info(f"\n{'=' * 60}")
        logger.info("BENCHMARK RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Model: {self.args.lm_type.upper()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch Size: {self.args.batch_size}")
        logger.info(f"Total Samples: {total_samples}")
        logger.info(f"\nLatency (per batch):")
        logger.info(f"  Average: {avg_latency:.2f} ms")
        logger.info(f"  P50: {p50_latency:.2f} ms")
        logger.info(f"  P95: {p95_latency:.2f} ms")
        logger.info(f"  P99: {p99_latency:.2f} ms")
        logger.info(f"\nThroughput:")
        logger.info(f"  Average: {avg_throughput:.2f} samples/sec")
        logger.info(f"  Total time: {sum(latencies) / 1000:.2f} sec")

        if torch.cuda.is_available():
            logger.info(f"\nGPU Memory:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            logger.info(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        logger.info(f"{'=' * 60}\n")

        # Save results
        if self.args.output_file:
            results = {
                'timestamp': datetime.now().isoformat(),
                'model': self.args.lm_type,
                'device': str(self.device),
                'batch_size': self.args.batch_size,
                'total_samples': total_samples,
                'avg_latency_ms': avg_latency,
                'p50_latency_ms': p50_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'avg_throughput_samples_per_sec': avg_throughput,
            }

            import json
            with open(self.args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {self.args.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark TabFormer inference')
    parser.add_argument('--lm_type', default='bert', choices=['bert', 'gpt2'],
                        help='Model type')
    parser.add_argument('--flatten', action='store_true',
                        help='Use flattened input')
    parser.add_argument('--field_ce', action='store_true',
                        help='Use field-wise cross entropy')
    parser.add_argument('--data_root', type=str, default='./data/credit_card/',
                        help='Data directory')
    parser.add_argument('--data_fname', type=str, default='card_transaction.v1',
                        help='Data file name')
    parser.add_argument('--data_extension', type=str, default='',
                        help='Data file extension')
    parser.add_argument('--model_dir', type=str, default='./checkpoints',
                        help='Model checkpoint directory')
    parser.add_argument('--nrows', type=int, default=10000,
                        help='Number of rows to load from dataset')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum samples to use for benchmarking')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--field_hs', type=int, default=64,
                        help='Field hidden size')
    parser.add_argument('--stride', type=int, default=5,
                        help='Stride for sliding window')
    parser.add_argument('--cached', action='store_true',
                        help='Use cached data')
    parser.add_argument('--skip_user', action='store_true',
                        help='Skip user field')
    parser.add_argument('--output_file', type=str, default='benchmark_results.json',
                        help='Output file for benchmark results')

    args = parser.parse_args()

    # Run benchmark
    benchmark = InferenceBenchmark(args)
    tokenizer = benchmark.load_model()
    benchmark.warmup(tokenizer, num_warmup=10)
    benchmark.benchmark(tokenizer)


if __name__ == '__main__':
    main()
