"""
Pure PyTorch BERT implementation compatible with torch.compile(fullgraph=True).

No HuggingFace transformers imports — avoids AttentionInterface dispatch,
@check_model_inputs decorator introspection, and dataclass return types
that cause graph breaks in torch.compile(fullgraph=True).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device
            )

        embeddings = (
            inputs_embeds
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NativeBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout_p = config.attention_probs_dropout_prob

    def _transpose_for_scores(self, x):
        # x: (batch, seq_len, all_head_size) -> (batch, heads, seq_len, head_size)
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None):
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(hidden_states))
        v = self._transpose_for_scores(self.value(hidden_states))

        dropout_p = self.dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout_p
        )
        # (batch, heads, seq_len, head_size) -> (batch, seq_len, all_head_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        bsz, seq_len, _, _ = attn_output.size()
        attn_output = attn_output.view(bsz, seq_len, self.all_head_size)
        return attn_output


class NativeBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NativeBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = NativeBertSelfAttention(config)
        self.attention_output = NativeBertSelfOutput(config)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.act_fn = F.gelu

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_output(attn_output, hidden_states)

        # FFN
        intermediate_output = self.act_fn(self.intermediate(attn_output))
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_LayerNorm(layer_output + attn_output)
        return layer_output


class NativeBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            [NativeBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class NativeBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = NativeBertEmbeddings(config)
        self.encoder = NativeBertEncoder(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        # Convert attention_mask from [batch, seq_len] to [batch, 1, 1, seq_len]
        # for broadcasting with (batch, heads, seq_len, seq_len)
        if attention_mask is not None:
            # 1.0 for tokens to attend, 0.0 for masked → convert to additive mask
            extended_attention_mask = attention_mask[:, None, None, :].to(
                dtype=embedding_output.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
                embedding_output.dtype
            ).min
        else:
            extended_attention_mask = None

        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        return (sequence_output,)
