"""
Verify parameter counts for TabFormer configs.
"""

def count_params_20M():
    """20M TabFormer with field_hidden_size=384."""
    print("\n" + "="*60)
    print("20M TabFormer Parameter Count")
    print("="*60)

    field_hidden_size = 384
    hidden_size = 384
    field_num_layers = 4
    seq_num_layers = 8
    field_ffn_size = 768
    seq_ffn_size = 768
    final_embedding_size = 256

    # Field embeddings (14 separate embedding layers)
    field_emb_0_1 = 2 * (300 + 2) * field_hidden_size  # 2 fields with 300 categories
    field_emb_2_3 = 2 * (30 + 2) * field_hidden_size   # 2 fields with 30 categories
    field_emb_4_5 = 2 * (200 + 2) * field_hidden_size  # 2 fields with 200 buckets
    field_emb_6_13 = 8 * (20 + 2) * field_hidden_size  # 8 fields with 20 buckets

    total_field_emb = field_emb_0_1 + field_emb_2_3 + field_emb_4_5 + field_emb_6_13
    print(f"Field embeddings: {total_field_emb:,} ({total_field_emb/1e6:.2f}M)")
    print(f"  - 2x Embedding(302, {field_hidden_size}): {field_emb_0_1:,}")
    print(f"  - 2x Embedding(32, {field_hidden_size}): {field_emb_2_3:,}")
    print(f"  - 2x Embedding(202, {field_hidden_size}): {field_emb_4_5:,}")
    print(f"  - 8x Embedding(22, {field_hidden_size}): {field_emb_6_13:,}")

    # Field transformer (4 layers)
    # Each layer: Q, K, V, O projections + 2 FFN layers + 2 RMSNorm
    field_qkvo = field_num_layers * 4 * (field_hidden_size * field_hidden_size)
    field_ffn = field_num_layers * 2 * (field_hidden_size * field_ffn_size)
    field_norm = field_num_layers * 2 * field_hidden_size  # RMSNorm weights
    total_field_transformer = field_qkvo + field_ffn + field_norm
    print(f"\nField transformer ({field_num_layers} layers): {total_field_transformer:,} ({total_field_transformer/1e6:.2f}M)")
    print(f"  - QKVO projections: {field_qkvo:,}")
    print(f"  - FFN: {field_ffn:,}")
    print(f"  - RMSNorm: {field_norm:,}")

    # Field projection
    field_proj = (field_hidden_size * 14) * hidden_size
    print(f"\nField projection: {field_proj:,} ({field_proj/1e6:.2f}M)")
    print(f"  - Linear({field_hidden_size * 14} → {hidden_size})")

    # Sequence transformer (8 layers, causal GPT with RoPE)
    seq_qkvo = seq_num_layers * 4 * (hidden_size * hidden_size)
    seq_ffn = seq_num_layers * 2 * (hidden_size * seq_ffn_size)
    seq_norm = (seq_num_layers * 2 + 1) * hidden_size  # 2 per layer + final norm
    # Note: RoPE has no learnable parameters (just cos/sin buffers)
    total_seq_transformer = seq_qkvo + seq_ffn + seq_norm
    print(f"\nSequence transformer ({seq_num_layers} layers): {total_seq_transformer:,} ({total_seq_transformer/1e6:.2f}M)")
    print(f"  - QKVO projections: {seq_qkvo:,}")
    print(f"  - FFN: {seq_ffn:,}")
    print(f"  - RMSNorm: {seq_norm:,}")
    print(f"  - RoPE: 0 (no learnable params)")

    # Output projection
    output_proj = hidden_size * final_embedding_size
    print(f"\nOutput projection: {output_proj:,} ({output_proj/1e6:.2f}M)")
    print(f"  - Linear({hidden_size} → {final_embedding_size})")

    # Total
    total = total_field_emb + total_field_transformer + field_proj + total_seq_transformer + output_proj
    print(f"\n{'='*60}")
    print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"{'='*60}")

    return total


def count_params_120M():
    """120M TabFormer with field_hidden_size=768."""
    print("\n" + "="*60)
    print("120M TabFormer Parameter Count")
    print("="*60)

    field_hidden_size = 768
    hidden_size = 768
    field_num_layers = 6
    seq_num_layers = 10
    field_ffn_size = 2048
    seq_ffn_size = 2048
    final_embedding_size = 512

    # Field embeddings
    field_emb_0_1 = 2 * (300 + 2) * field_hidden_size
    field_emb_2_3 = 2 * (30 + 2) * field_hidden_size
    field_emb_4_5 = 2 * (200 + 2) * field_hidden_size
    field_emb_6_13 = 8 * (20 + 2) * field_hidden_size

    total_field_emb = field_emb_0_1 + field_emb_2_3 + field_emb_4_5 + field_emb_6_13
    print(f"Field embeddings: {total_field_emb:,} ({total_field_emb/1e6:.2f}M)")

    # Field transformer
    field_qkvo = field_num_layers * 4 * (field_hidden_size * field_hidden_size)
    field_ffn = field_num_layers * 2 * (field_hidden_size * field_ffn_size)
    field_norm = field_num_layers * 2 * field_hidden_size
    total_field_transformer = field_qkvo + field_ffn + field_norm
    print(f"Field transformer ({field_num_layers} layers): {total_field_transformer:,} ({total_field_transformer/1e6:.2f}M)")

    # Field projection
    field_proj = (field_hidden_size * 14) * hidden_size
    print(f"Field projection: {field_proj:,} ({field_proj/1e6:.2f}M)")

    # Sequence transformer
    seq_qkvo = seq_num_layers * 4 * (hidden_size * hidden_size)
    seq_ffn = seq_num_layers * 2 * (hidden_size * seq_ffn_size)
    seq_norm = (seq_num_layers * 2 + 1) * hidden_size
    total_seq_transformer = seq_qkvo + seq_ffn + seq_norm
    print(f"Sequence transformer ({seq_num_layers} layers): {total_seq_transformer:,} ({total_seq_transformer/1e6:.2f}M)")

    # Output projection
    output_proj = hidden_size * final_embedding_size
    print(f"Output projection: {output_proj:,} ({output_proj/1e6:.2f}M)")

    # Total
    total = total_field_emb + total_field_transformer + field_proj + total_seq_transformer + output_proj
    print(f"\n{'='*60}")
    print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"{'='*60}")

    return total


def count_params_720M():
    """720M TabFormer with field_hidden_size=1536."""
    print("\n" + "="*60)
    print("720M TabFormer Parameter Count")
    print("="*60)

    field_hidden_size = 1536
    hidden_size = 1536
    field_num_layers = 6
    seq_num_layers = 18
    field_ffn_size = 4096
    seq_ffn_size = 4096
    final_embedding_size = 1024

    # Field embeddings
    field_emb_0_1 = 2 * (300 + 2) * field_hidden_size
    field_emb_2_3 = 2 * (30 + 2) * field_hidden_size
    field_emb_4_5 = 2 * (200 + 2) * field_hidden_size
    field_emb_6_13 = 8 * (20 + 2) * field_hidden_size

    total_field_emb = field_emb_0_1 + field_emb_2_3 + field_emb_4_5 + field_emb_6_13
    print(f"Field embeddings: {total_field_emb:,} ({total_field_emb/1e6:.2f}M)")

    # Field transformer
    field_qkvo = field_num_layers * 4 * (field_hidden_size * field_hidden_size)
    field_ffn = field_num_layers * 2 * (field_hidden_size * field_ffn_size)
    field_norm = field_num_layers * 2 * field_hidden_size
    total_field_transformer = field_qkvo + field_ffn + field_norm
    print(f"Field transformer ({field_num_layers} layers): {total_field_transformer:,} ({total_field_transformer/1e6:.2f}M)")

    # Field projection
    field_proj = (field_hidden_size * 14) * hidden_size
    print(f"Field projection: {field_proj:,} ({field_proj/1e6:.2f}M)")

    # Sequence transformer
    seq_qkvo = seq_num_layers * 4 * (hidden_size * hidden_size)
    seq_ffn = seq_num_layers * 2 * (hidden_size * seq_ffn_size)
    seq_norm = (seq_num_layers * 2 + 1) * hidden_size
    total_seq_transformer = seq_qkvo + seq_ffn + seq_norm
    print(f"Sequence transformer ({seq_num_layers} layers): {total_seq_transformer:,} ({total_seq_transformer/1e6:.2f}M)")

    # Output projection
    output_proj = hidden_size * final_embedding_size
    print(f"Output projection: {output_proj:,} ({output_proj/1e6:.2f}M)")

    # Total
    total = total_field_emb + total_field_transformer + field_proj + total_seq_transformer + output_proj
    print(f"\n{'='*60}")
    print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"{'='*60}")

    return total


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TabFormer Parameter Verification")
    print("With field_hidden_size = hidden_size")
    print("="*60)

    count_params_20M()
    count_params_120M()
    count_params_720M()
