"""
Pure PyTorch GPT implementation compatible with torch.compile(fullgraph=True).

Features:
- Causal (autoregressive) self-attention
- RoPE (Rotary Position Embeddings) with FP32 cos/sin tables
- RMSNorm with FP32 computation
- Pre-norm architecture (modern GPT style)
- No HuggingFace dependencies to avoid graph breaks

Precision requirements:
- FP32: RMSNorm computation, Softmax (in attention), RoPE cos/sin tables
- BF16: Linear projections (Q/K/V, FFN), matmuls, embedding lookups
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization in FP32 for numerical stability."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _apply(self, fn):
        """Override _apply to keep weight in FP32 when model.bfloat16() is called."""
        super()._apply(fn)
        # Force weight back to FP32 after any dtype conversion
        if hasattr(self, 'weight'):
            self.weight.data = self.weight.data.float()
        return self

    def forward(self, x):
        # Always compute in FP32 for numerical stability
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # Convert back to input dtype and apply learnable weight (FP32)
        return (self.weight * x).to(input_dtype)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from Su et al. (2021).
    Cos/sin tables are computed and stored in FP32 for precision.
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute cos/sin tables in FP32
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)

        # Store in FP32
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _apply(self, fn):
        """Override _apply to keep cos/sin buffers in FP32 when model.bfloat16() is called."""
        super()._apply(fn)
        # Force cos/sin buffers back to FP32 after any dtype conversion
        if hasattr(self, 'cos_cached'):
            self.cos_cached = self.cos_cached.float()
        if hasattr(self, 'sin_cached'):
            self.sin_cached = self.sin_cached.float()
        if hasattr(self, 'inv_freq'):
            self.inv_freq = self.inv_freq.float()
        return self

    def forward(self, q, k):
        """
        Apply RoPE to query and key tensors.
        Args:
            q: (batch, heads, seq_len, head_dim)
            k: (batch, heads, seq_len, head_dim)
        Returns:
            q_rot, k_rot: Rotated query and key tensors
        """
        seq_len = q.shape[2]
        input_dtype = q.dtype

        # Get cos/sin for current sequence length (FP32 for precision)
        cos = self.cos_cached[:seq_len, :self.dim]  # (seq_len, dim)
        sin = self.sin_cached[:seq_len, :self.dim]  # (seq_len, dim)

        # Reshape for broadcasting: (1, 1, seq_len, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation (computation happens in FP32 due to cos/sin)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        # Convert back to input dtype (BF16) to match V for attention
        q_rot = q_rot.to(input_dtype)
        k_rot = k_rot.to(input_dtype)

        return q_rot, k_rot

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


class NativeGPTEmbeddings(nn.Module):
    """
    GPT embeddings: token embeddings only (no position embeddings - using RoPE instead).
    """
    def __init__(self, config):
        super().__init__()
        # Embedding will run in BF16 automatically when using autocast
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = self.dropout(inputs_embeds)
        return embeddings


class NativeGPTCausalSelfAttention(nn.Module):
    """
    Causal self-attention with RoPE.
    Softmax operates in FP32 for numerical stability.
    """
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections (will run in BF16)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # RoPE with FP32 cos/sin tables
        self.rope = RotaryPositionEmbedding(
            dim=self.attention_head_size,
            max_seq_len=config.max_position_embeddings,
            base=10000
        )

        self.dropout_p = config.attention_probs_dropout_prob

    def _transpose_for_scores(self, x):
        # x: (batch, seq_len, all_head_size) -> (batch, heads, seq_len, head_size)
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None):
        # Project to Q, K, V (BF16 computation)
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(hidden_states))
        v = self._transpose_for_scores(self.value(hidden_states))

        # Apply RoPE to Q and K (cos/sin tables are FP32)
        q, k = self.rope(q, k)

        # Causal attention with scaled_dot_product_attention
        # Softmax automatically runs in FP32 for stability
        dropout_p = self.dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=(attention_mask is None)  # Use causal mask if no mask provided
        )

        # (batch, heads, seq_len, head_size) -> (batch, seq_len, all_head_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        bsz, seq_len, _, _ = attn_output.size()
        attn_output = attn_output.view(bsz, seq_len, self.all_head_size)

        return attn_output


class NativeGPTSelfOutput(nn.Module):
    """Output projection after attention."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class NativeGPTBlock(nn.Module):
    """
    GPT transformer block with pre-norm architecture.

    Architecture:
        x = x + attn(RMSNorm(x))
        x = x + ffn(RMSNorm(x))
    """
    def __init__(self, config):
        super().__init__()
        # Pre-norm: normalize before attention
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = NativeGPTCausalSelfAttention(config)
        self.attention_output = NativeGPTSelfOutput(config)

        # Pre-norm: normalize before FFN
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.mlp_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.mlp_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.act_fn = F.gelu

    def forward(self, hidden_states, attention_mask=None):
        # Pre-norm attention block with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)  # RMSNorm in FP32
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_output(attn_output)
        hidden_states = residual + attn_output

        # Pre-norm FFN block with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)  # RMSNorm in FP32
        hidden_states = self.mlp_fc(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.mlp_proj(hidden_states)
        hidden_states = self.mlp_dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NativeGPTEncoder(nn.Module):
    """Stack of GPT transformer blocks."""
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [NativeGPTBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class NativeGPTModel(nn.Module):
    """
    Native PyTorch GPT model with:
    - Causal self-attention
    - RoPE position embeddings
    - RMSNorm
    - Pre-norm architecture
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = NativeGPTEmbeddings(config)
        self.encoder = NativeGPTEncoder(config)
        # Final layer norm (pre-norm style)
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
    ):
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        # Convert attention_mask to causal mask format if provided
        # For padding: 1.0 for tokens to attend, 0.0 for masked
        if attention_mask is not None:
            # Convert to additive mask for scaled_dot_product_attention
            # Shape: (batch, 1, 1, seq_len) for broadcasting
            extended_attention_mask = attention_mask[:, None, None, :].to(
                dtype=embedding_output.dtype
            )
            # Convert: 1.0 -> 0.0 (attend), 0.0 -> -inf (mask)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
                embedding_output.dtype
            ).min
        else:
            extended_attention_mask = None

        # Pass through transformer blocks
        sequence_output = self.encoder(embedding_output, extended_attention_mask)

        # Final layer norm
        sequence_output = self.ln_f(sequence_output)

        return (sequence_output,)


class NativeGPTLMHeadModel(nn.Module):
    """
    GPT Language Model with causal LM head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = NativeGPTModel(config)
        # LM head for next token prediction
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Optionally tie weights with input embeddings
        if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # Get hidden states from transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        # Compute logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift labels for causal LM: predict next token
            # logits: [..., :-1, :] predicts labels: [..., 1:]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for loss computation
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {
            'loss': loss,
            'logits': lm_logits,
            'hidden_states': hidden_states,
        }
