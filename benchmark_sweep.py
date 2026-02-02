"""
Comprehensive benchmark sweep for TabFormer models.
Benchmarks both BERT and GPT2 with 120M and 20M configs across batch sizes.
"""
import os
import logging
import torch
import time
import json
import argparse
from datetime import datetime

from transformers import BertTokenizer, GPT2Config
from models.modules import TabFormerBertLM, TabFormerGPT2
from models.tabformer_tokenizer import TabFormerTokenizer
from dataset.vocab import Vocabulary
from model_configs import get_model_config, estimate_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class BenchmarkSweep:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []

    def create_dummy_vocab(self, vocab_size=30522):
        """Create a dummy vocabulary for benchmarking."""
        from misc.utils import ddict
        import tempfile

        class DummyVocab(ddict):
            def __len__(self):
                return self.vocab_size

        vocab = DummyVocab()
        vocab.vocab_size = vocab_size
        vocab.pad_token = '[PAD]'
        vocab.mask_token = '[MASK]'
        vocab.unk_token = '[UNK]'
        vocab.bos_token = '[CLS]'
        vocab.eos_token = '[SEP]'

        # Create a temporary vocab file for BertTokenizer
        vocab_file = os.path.join(tempfile.gettempdir(), 'benchmark_vocab.txt')
        if not os.path.exists(vocab_file):
            special = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
            with open(vocab_file, 'w') as f:
                for token in special:
                    f.write(token + '\n')
                for i in range(vocab_size - len(special)):
                    f.write(f'token_{i}\n')
        vocab.filename = vocab_file

        # Add dummy field structure for TabFormer
        vocab.field_keys = ['field_' + str(i) for i in range(12)]
        vocab.adap_sm_cols = set()

        # Mock methods
        vocab.get_field_keys = lambda remove_target=False, ignore_special=False: vocab.field_keys
        vocab.get_field_ids = lambda field_name: list(range(100))
        vocab.get_from_global_ids = lambda global_ids, what_to_get: global_ids
        vocab.get_special_tokens = lambda: {
            'pad_token': '[PAD]',
            'mask_token': '[MASK]',
            'unk_token': '[UNK]',
            'bos_token': '[CLS]',
            'eos_token': '[SEP]'
        }

        return vocab

    def create_model(self, model_type, config_name):
        """Create a TabFormer model with specified config."""
        config = get_model_config(config_name)
        vocab = self.create_dummy_vocab()
        special_tokens = vocab.get_special_tokens()

        logger.info(f"Creating {model_type} model with {config_name} config")
        logger.info(f"Config: {config}")

        if model_type == 'bert':
            model = TabFormerBertLM(
                special_tokens=special_tokens,
                vocab=vocab,
                field_ce=True,
                flatten=False,
                ncols=config['ncols'],
                field_hidden_size=config['field_hidden_size']
            )
            # Override config with our custom settings
            model.config.num_hidden_layers = config['num_hidden_layers']
            model.config.hidden_size = config['hidden_size']
            model.config.intermediate_size = config['intermediate_size']
            model.config.num_attention_heads = config['num_attention_heads']
            model.config._attn_implementation = self.args.attn_impl

            # Reinitialize model with new config (hierarchical wrapper handles 3D input)
            from models.modules import TabFormerHierarchicalLM
            model.model = TabFormerHierarchicalLM(model.config, vocab)
            tokenizer = model.tokenizer

        else:  # gpt2
            gpt_config = GPT2Config(
                vocab_size=30522,
                n_positions=1024,
                n_embd=config['hidden_size'],
                n_layer=config['num_hidden_layers'],
                n_head=config['num_attention_heads'],
                n_inner=config['intermediate_size'],
                attn_implementation=self.args.attn_impl
            )

            from models.tabformer_gpt2 import TabFormerGPT2LMHeadModel
            model_gpt = TabFormerGPT2LMHeadModel(gpt_config, vocab)
            tokenizer = TabFormerTokenizer(
                unk_token=special_tokens['unk_token'],
                bos_token=special_tokens['bos_token'],
                eos_token=special_tokens['eos_token']
            )
            model = type('obj', (object,), {'model': model_gpt, 'tokenizer': tokenizer})()

        model.model = model.model.to(self.device)
        if self.args.attn_impl in ('flash_attention_2', 'sdpa'):
            model.model = model.model.to(torch.float16)
        model.model.eval()

        if self.args.torch_compile:
            logger.info("Wrapping model with torch.compile")
            model.model = torch.compile(model.model)

        # Count actual parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        logger.info(f"Actual model parameters: {total_params / 1e6:.1f}M")

        return model, tokenizer

    def create_dummy_batch(self, batch_size, seq_len=128, ncols=12):
        """Create dummy input data."""
        if self.args.flatten:
            # Flattened: (batch_size, seq_len)
            input_ids = torch.randint(0, 30000, (batch_size, seq_len)).to(self.device)
        else:
            # Hierarchical: (batch_size, seq_len, ncols)
            input_ids = torch.randint(0, 30000, (batch_size, seq_len, ncols)).to(self.device)

        return {'input_ids': input_ids}

    def precompile_shapes(self, model, batch_sizes, seq_lens):
        """Pre-compile all (batch_size, seq_len) shapes so AOT compilation
        doesn't pollute benchmark measurements."""
        shapes = [(bs, sl) for bs in batch_sizes for sl in seq_lens]
        logger.info(f"Pre-compiling {len(shapes)} shapes...")
        for i, (bs, sl) in enumerate(shapes):
            logger.info(f"  Compiling shape {i+1}/{len(shapes)}: batch_size={bs}, seq_len={sl}")
            dummy_batch = self.create_dummy_batch(bs, seq_len=sl)
            with torch.no_grad():
                _ = model.model(**dummy_batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        logger.info("Pre-compilation complete.")

    def benchmark_config(self, model, model_type, config_name, batch_size, seq_len):
        """Benchmark a single configuration using a pre-compiled model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_type.upper()} | {config_name} | batch_size={batch_size} | seq_len={seq_len}")
        logger.info(f"{'='*60}")

        # Warmup (model already compiled, this just warms caches)
        logger.info("Warming up...")
        dummy_batch = self.create_dummy_batch(batch_size, seq_len=seq_len)
        for _ in range(self.args.warmup_iterations):
            with torch.no_grad():
                _ = model.model(**dummy_batch)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Benchmark
        logger.info(f"Running {self.args.num_iterations} iterations...")
        latencies = []

        for i in range(self.args.num_iterations):
            dummy_batch = self.create_dummy_batch(batch_size, seq_len=seq_len)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            with torch.no_grad():
                _ = model.model(**dummy_batch)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 10 == 0:
                logger.info(f"  Iteration {i+1}/{self.args.num_iterations}: {latency_ms:.2f}ms")

        # Calculate statistics
        latencies = latencies[5:]  # Drop first 5 for stability
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        throughput = (batch_size * 1000) / avg_latency

        result = {
            'model_type': model_type,
            'config_name': config_name,
            'attn_impl': self.args.attn_impl,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'avg_latency_ms': round(avg_latency, 2),
            'min_latency_ms': round(min_latency, 2),
            'max_latency_ms': round(max_latency, 2),
            'p50_latency_ms': round(p50_latency, 2),
            'p99_latency_ms': round(p99_latency, 2),
            'throughput_samples_per_sec': round(throughput, 2),
            'torch_compile': self.args.torch_compile,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"\nResults:")
        logger.info(f"  Avg Latency: {avg_latency:.2f}ms")
        logger.info(f"  P50 Latency: {p50_latency:.2f}ms")
        logger.info(f"  P99 Latency: {p99_latency:.2f}ms")
        logger.info(f"  Throughput: {throughput:.2f} samples/sec")

        self.results.append(result)

    def run_sweep(self):
        """Run full benchmark sweep."""
        model_types = self.args.model_types.split(',')
        config_names = self.args.config_names.split(',')
        batch_sizes = [int(b) for b in self.args.batch_sizes.split(',')]
        seq_lens = [int(s) for s in self.args.seq_lens.split(',')]

        logger.info(f"Starting benchmark sweep:")
        logger.info(f"  Model types: {model_types}")
        logger.info(f"  Configs: {config_names}")
        logger.info(f"  Batch sizes: {batch_sizes}")
        logger.info(f"  Sequence lengths: {seq_lens}")
        logger.info(f"  Attention impl: {self.args.attn_impl}")
        logger.info(f"  torch.compile: {self.args.torch_compile}")
        logger.info(f"  Iterations per config: {self.args.num_iterations}")

        for model_type in model_types:
            for config_name in config_names:
                try:
                    model, tokenizer = self.create_model(model_type, config_name)

                    # Pre-compile all shapes before benchmarking
                    if self.args.torch_compile:
                        self.precompile_shapes(model, batch_sizes, seq_lens)

                    for batch_size in batch_sizes:
                        for seq_len in seq_lens:
                            try:
                                self.benchmark_config(model, model_type, config_name, batch_size, seq_len)
                            except Exception as e:
                                logger.error(f"Error benchmarking {model_type}/{config_name}/bs={batch_size}/seq={seq_len}: {e}")
                                import traceback
                                traceback.print_exc()

                    # Cleanup after all shapes for this model
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception as e:
                    logger.error(f"Error creating model {model_type}/{config_name}: {e}")
                    import traceback
                    traceback.print_exc()

        # Save results
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save benchmark results to JSON."""
        output_file = self.args.output_file
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nResults saved to: {output_file}")

    def print_summary(self):
        """Print summary table of results."""
        logger.info(f"\n{'='*110}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'='*110}")
        logger.info(f"{'Model':<8} {'Config':<8} {'Attn':<18} {'Compile':<9} {'BS':<4} {'SeqLen':<7} {'Avg(ms)':<10} {'P99(ms)':<10} {'Throughput':<15}")
        logger.info(f"{'-'*110}")

        for r in self.results:
            compile_str = 'yes' if r['torch_compile'] else 'no'
            logger.info(
                f"{r['model_type']:<8} {r['config_name']:<8} {r['attn_impl']:<18} {compile_str:<9} {r['batch_size']:<4} "
                f"{r['seq_len']:<7} {r['avg_latency_ms']:<10.2f} {r['p99_latency_ms']:<10.2f} "
                f"{r['throughput_samples_per_sec']:<15.2f}"
            )


def main():
    parser = argparse.ArgumentParser(description='TabFormer Benchmark Sweep')
    parser.add_argument('--model_types', type=str, default='bert,gpt2',
                        help='Comma-separated model types (bert,gpt2)')
    parser.add_argument('--config_names', type=str, default='120M,20M',
                        help='Comma-separated config names (120M,20M)')
    parser.add_argument('--batch_sizes', type=str, default='1,4,16,32',
                        help='Comma-separated batch sizes')
    parser.add_argument('--seq_lens', type=str, default='10,50,100',
                        help='Comma-separated sequence lengths')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of iterations per config')
    parser.add_argument('--warmup_iterations', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--flatten', action='store_true',
                        help='Use flattened input (for GPT2)')
    parser.add_argument('--output_file', type=str, default='benchmark_sweep_results.json',
                        help='Output file for results')
    parser.add_argument('--attn_impl', type=str, default='eager',
                        choices=['eager', 'sdpa', 'flash_attention_2'],
                        help='Attention implementation (eager, sdpa, flash_attention_2)')
    parser.add_argument('--torch_compile', action='store_true',
                        help='Wrap model with torch.compile')

    args = parser.parse_args()

    benchmark = BenchmarkSweep(args)
    benchmark.run_sweep()


if __name__ == '__main__':
    main()
