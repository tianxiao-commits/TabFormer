"""
Minimal test: SDPA + torch.compile(fullgraph=True) on TabFormer BERT.
Verifies native PyTorch BERT passes fullgraph=True compilation.
"""
import os
import torch
import torch._dynamo
import logging
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from misc.utils import ddict
from models.modules import TabFormerBertLM, TabFormerHierarchicalLM
from model_configs import get_model_config


def create_dummy_vocab(vocab_size=30522):
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

    vocab_file = os.path.join(tempfile.gettempdir(), 'test_vocab.txt')
    special = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    with open(vocab_file, 'w') as f:
        for token in special:
            f.write(token + '\n')
        for i in range(vocab_size - len(special)):
            f.write(f'token_{i}\n')
    vocab.filename = vocab_file

    vocab.field_keys = ['field_' + str(i) for i in range(12)]
    vocab.adap_sm_cols = set()
    vocab.get_field_keys = lambda remove_target=False, ignore_special=False: vocab.field_keys
    vocab.get_field_ids = lambda field_name: list(range(100))
    vocab.get_from_global_ids = lambda global_ids, what_to_get: global_ids
    vocab.get_special_tokens = lambda: {
        'pad_token': '[PAD]', 'mask_token': '[MASK]',
        'unk_token': '[UNK]', 'bos_token': '[CLS]', 'eos_token': '[SEP]'
    }
    return vocab


def create_model(config_name='20M'):
    config = get_model_config(config_name)
    vocab = create_dummy_vocab()
    special_tokens = vocab.get_special_tokens()

    model = TabFormerBertLM(
        special_tokens=special_tokens, vocab=vocab,
        field_ce=True, flatten=False,
        ncols=config['ncols'], field_hidden_size=config['field_hidden_size']
    )
    model.config.num_hidden_layers = config['num_hidden_layers']
    model.config.hidden_size = config['hidden_size']
    model.config.intermediate_size = config['intermediate_size']
    model.config.num_attention_heads = config['num_attention_heads']

    model.model = TabFormerHierarchicalLM(model.config, vocab)
    return model


def test_fullgraph():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Device: {device}, dtype: {dtype}")

    # Dummy input: (batch, seq_len, ncols)
    input_ids = torch.randint(0, 30000, (2, 10, 12)).to(device)

    # --- Test: torch.compile(fullgraph=True) on fresh model ---
    torch._dynamo.reset()
    logger.info("=== Testing torch.compile(fullgraph=True) on TabFormerHierarchicalLM ===")
    model = create_model(config_name='20M')
    model.model = model.model.to(device).to(dtype).eval()
    try:
        compiled_model = torch.compile(model.model, fullgraph=True)
        with torch.no_grad():
            out = compiled_model(input_ids=input_ids)
        logger.info(f"PASSED. Output type: {type(out)}")
    except Exception as e:
        logger.error(f"FAILED: {e}")
    del model
    torch.cuda.empty_cache()

    # --- Test sub-components ---
    torch._dynamo.reset()
    logger.info("\n=== Testing TabFormerEmbeddings with fullgraph=True ===")
    model = create_model(config_name='20M')
    model.model = model.model.to(device).to(dtype).eval()
    try:
        compiled_emb = torch.compile(model.model.tab_embeddings, fullgraph=True)
        with torch.no_grad():
            emb_out = compiled_emb(input_ids)
        logger.info(f"TabFormerEmbeddings PASSED. Shape: {emb_out.shape}")
    except Exception as e:
        logger.error(f"TabFormerEmbeddings FAILED: {e}")

    torch._dynamo.reset()
    logger.info("\n=== Testing TabFormerBertForMaskedLM with fullgraph=True ===")
    try:
        with torch.no_grad():
            embeds = model.model.tab_embeddings(input_ids)
        compiled_bert = torch.compile(model.model.tb_model, fullgraph=True)
        with torch.no_grad():
            bert_out = compiled_bert(inputs_embeds=embeds)
        logger.info(f"TabFormerBertForMaskedLM PASSED.")
    except Exception as e:
        logger.error(f"TabFormerBertForMaskedLM FAILED: {e}")

    torch._dynamo.reset()
    logger.info("\n=== Testing NativeBertModel (encoder only) with fullgraph=True ===")
    try:
        compiled_enc = torch.compile(model.model.tb_model.bert, fullgraph=True)
        with torch.no_grad():
            enc_out = compiled_enc(inputs_embeds=embeds)
        logger.info(f"NativeBertModel PASSED.")
    except Exception as e:
        logger.error(f"NativeBertModel FAILED: {e}")

    logger.info("\n=== Done ===")


if __name__ == '__main__':
    test_fullgraph()
