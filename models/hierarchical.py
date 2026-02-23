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


class NativeFieldTransformerBlock(nn.Module):
    """
    Bidirectional transformer block for field-level processing.
    Similar to BERT but with RMSNorm and torch.compile compatibility.

    Uses pre-norm architecture with RMSNorm (FP32 computation).
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Pre-norm
        self.norm1 = RMSNorm(d_model)

        # Multi-head attention (bidirectional)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        # Pre-norm for FFN
        self.norm2 = RMSNorm(d_model)

        # Feed-forward network
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """Split into multiple attention heads."""
        # x: (batch, seq_len, d_model) -> (batch, nhead, seq_len, head_dim)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.nhead, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: (batch, seq_len, d_model) or (seq_len, batch, d_model)
            src_mask: Optional attention mask
        """
        # Handle both (seq, batch, dim) and (batch, seq, dim) formats
        is_batched = src.dim() == 3
        if not is_batched:
            src = src.unsqueeze(1)

        # Check if input is (seq, batch, dim) and transpose to (batch, seq, dim)
        if src.size(0) != src.size(1) and src.size(2) == self.d_model:
            # Likely (seq, batch, dim) format from nn.TransformerEncoder
            needs_transpose = True
            src = src.transpose(0, 1)
        else:
            needs_transpose = False

        # Pre-norm + Self-attention with residual
        residual = src
        src_norm = self.norm1(src)  # RMSNorm in FP32

        # Multi-head attention
        q = self._split_heads(self.q_proj(src_norm))
        k = self._split_heads(self.k_proj(src_norm))
        v = self._split_heads(self.v_proj(src_norm))

        # Scaled dot-product attention (bidirectional, softmax in FP32)
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=src_mask,
            dropout_p=dropout_p,
            is_causal=False  # Bidirectional attention
        )

        # Concatenate heads
        batch_size, _, seq_len, _ = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        src = residual + attn_output

        # Pre-norm + FFN with residual
        residual = src
        src_norm = self.norm2(src)  # RMSNorm in FP32

        ffn_output = self.fc1(src_norm)
        ffn_output = F.gelu(ffn_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.fc2(ffn_output)
        ffn_output = self.dropout(ffn_output)
        src = residual + ffn_output

        # Transpose back if needed
        if needs_transpose:
            src = src.transpose(0, 1)

        if not is_batched:
            src = src.squeeze(1)

        return src


class NativeFieldTransformerEncoder(nn.Module):
    """
    Stack of native field transformer blocks.
    Replacement for nn.TransformerEncoder with RMSNorm and torch.compile compatibility.
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            NativeFieldTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None):
        """
        Args:
            src: (seq_len, batch, d_model) or (batch, seq_len, d_model)
            mask: Optional attention mask
        """
        for layer in self.layers:
            src = layer(src, mask)
        return src


class TabFormerConcatEmbeddings(nn.Module):
    """TabFormerConcatEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabolary that is common across columns
               - `sparse=True` in `nn.Embedding` speeds up gradient computation for large vocabs

        Args:
            config.ncols
            config.vocab_size
            config.hidden_size

        Inputs:
            - **input_ids** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output'**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)
        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

        self.hidden_size = config.hidden_size
        self.field_hidden_size = config.field_hidden_size

    def forward(self, input_ids):
        input_shape = input_ids.size()

        embeds_sz = list(input_shape[:-1]) + [input_shape[-1] * self.field_hidden_size]
        inputs_embeds = self.lin_proj(self.word_embeddings(input_ids).view(embeds_sz))

        return inputs_embeds


class TabFormerEmbeddings(nn.Module):
    """TabFormerEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabolary that is common across columns

        Args:
            config.ncols
            config.num_layers (int): Number of transformer layers
            config.vocab_size
            config.hidden_size
            config.field_hidden_size
            config.native_field_transformer (bool): Use native field transformer with RMSNorm

        Inputs:
            - **input** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()

        if not hasattr(config, 'num_layers'):
            config.num_layers = 1
        if not hasattr(config, 'nhead'):
            config.nhead = 8
        if not hasattr(config, 'native_field_transformer'):
            config.native_field_transformer = False

        self.native_field_transformer = config.native_field_transformer

        self.word_embeddings = nn.Embedding(config.vocab_size, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)

        if self.native_field_transformer:
            # Use native field transformer with RMSNorm and torch.compile compatibility
            dropout = getattr(config, 'hidden_dropout_prob', 0.1)
            self.transformer_encoder = NativeFieldTransformerEncoder(
                d_model=config.field_hidden_size,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dim_feedforward=config.field_hidden_size,
                dropout=dropout
            )
        else:
            # Use PyTorch nn.TransformerEncoder (backward compatibility)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.field_hidden_size,
                nhead=config.nhead,
                dim_feedforward=config.field_hidden_size
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

    def forward(self, input_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        embeds_shape = list(inputs_embeds.size())

        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:])
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2]+[-1])

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds
