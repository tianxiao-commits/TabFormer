# Tabular Transformers for Modeling Multivariate Time Series

This repository provides the pytorch source code, and data for tabular transformers (TabFormer). Details are described in the paper [Tabular Transformers for Modeling Multivariate Time Series](http://arxiv.org/abs/2011.01843 ), to be presented at ICASSP 2021.

#### Summary
* Modules for hierarchical transformers for tabular data
* A synthetic credit card transaction dataset
* Modified Adaptive Softmax for handling masking
* Modified _DataCollatorForLanguageModeling_ for tabular data
* The modules are built within transformers from HuggingFace 🤗. (HuggingFace is ❤️)
---
### Requirements
* Python (3.7)
* Pytorch (1.6.0)
* HuggingFace / Transformer (3.2.0)
* scikit-learn (0.23.2)
* Pandas (1.1.2)

(X) represents the versions which code is tested on.

These can be installed using yaml by running : 
```
conda env create -f setup.yml
```
---

### Credit Card Transaction Dataset

The synthetic credit card transaction dataset is provided in [./data/credit_card](/data/credit_card/). There are 24M records with 12 fields.
You would need git-lfs to access the data. If you are facing issue related to LFS bandwidth, you can use this [direct link](https://ibm.box.com/v/tabformer-data) to access the data. You can then ignore git-lfs files by prefixing `GIT_LFS_SKIP_SMUDGE=1` to the `git clone ..` command.

![figure](./misc/cc_trans_dataset.png)

---

### PRSA Dataset
For PRSA dataset, one have to download the PRSA dataset from [Kaggle](https://www.kaggle.com/sid321axn/beijing-multisite-airquality-data-set) and place them in [./data/card](/data/card/) directory.

---

### Tabular BERT
To train a tabular BERT model on credit card transaction or PRSA dataset run :
```
$ python main.py --do_train --mlm --field_ce --lm_type bert \
                 --field_hs 64 --data_type [prsa/card] \
                 --output_dir [output_dir]
```


### Tabular GPT2
To train a tabular GPT2 model on credit card transactions for a particular _user-id_ :
```

$ python main.py --do_train --lm_type gpt2 --field_ce --flatten --data_type card \
                 --data_root [path_to_data] --user_ids [user-id] \
                 --output_dir [output_dir]
    
```

Description of some options (more can be found in _`args.py`_):
* `--data_type` choices are `prsa` and `card` for Beijing PM2.5 dataset and credit-card transaction dataset respecitively. 
* `--mlm` for masked language model; option for transformer trainer for BERT
* `--field_hs` hidden size for field level transformer
* `--lm_type` choices from `bert` and `gpt2`
* `--user_ids` option to pick only transacations from particular user ids.
---

### Benchmark Sweep

`benchmark_sweep.py` measures inference latency and throughput across model types, configs, batch sizes, and sequence lengths.

#### Basic usage

```bash
python benchmark_sweep.py \
  --model_types bert --config_names 20M \
  --batch_sizes 4 --seq_lens 10 \
  --num_iterations 100 --warmup_iterations 10
```

#### SDPA attention

Use `--attn_impl sdpa` to enable PyTorch's scaled dot-product attention, which automatically selects the best available kernel (FlashAttention, memory-efficient, or math):

```bash
python benchmark_sweep.py \
  --model_types bert --config_names 20M \
  --batch_sizes 4 --seq_lens 10 \
  --attn_impl sdpa
```

#### torch.compile

Add `--torch_compile` to wrap the model with `torch.compile`, which traces and optimizes the computation graph. The first few warmup iterations will be slower due to compilation, but subsequent iterations benefit from fused kernels and reduced overhead:

```bash
python benchmark_sweep.py \
  --model_types bert --config_names 20M \
  --batch_sizes 4 --seq_lens 10 \
  --attn_impl sdpa --torch_compile
```

`--torch_compile` works best with `--attn_impl sdpa` or `--attn_impl eager`. Flash Attention 2 uses a third-party CUDA kernel that causes graph breaks during compilation, limiting the benefit.

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model_types` | Comma-separated model types (`bert`, `gpt2`) | `bert,gpt2` |
| `--config_names` | Comma-separated configs (`120M`, `20M`) | `120M,20M` |
| `--batch_sizes` | Comma-separated batch sizes | `1,4,16,32` |
| `--seq_lens` | Comma-separated sequence lengths | `10,50,100` |
| `--num_iterations` | Benchmark iterations per config | `100` |
| `--warmup_iterations` | Warmup iterations (not measured) | `10` |
| `--attn_impl` | Attention implementation (`eager`, `sdpa`, `flash_attention_2`) | `eager` |
| `--torch_compile` | Wrap model with `torch.compile` | off |
| `--flatten` | Use flattened input (for GPT2) | off |
| `--output_file` | JSON output path | `benchmark_sweep_results.json` |

---

### Citation

```
@inproceedings{padhi2021tabular,
  title={Tabular transformers for modeling multivariate time series},
  author={Padhi, Inkit and Schiff, Yair and Melnyk, Igor and Rigotti, Mattia and Mroueh, Youssef and Dognin, Pierre and Ross, Jerret and Nair, Ravi and Altman, Erik},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3565--3569},
  year={2021},
  organization={IEEE},
  url={https://ieeexplore.ieee.org/document/9414142}
}
```
