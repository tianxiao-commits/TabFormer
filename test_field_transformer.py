"""
Test script to verify the native field transformer implementation.

This script:
1. Tests the native field transformer block
2. Verifies bidirectional attention (not causal)
3. Verifies RMSNorm usage
4. Tests TabFormerEmbeddings with native field transformer
5. Compares output shapes with original nn.TransformerEncoder
"""

import torch
from models.hierarchical import (
    NativeFieldTransformerBlock,
    NativeFieldTransformerEncoder,
    TabFormerEmbeddings,
    RMSNorm
)


class FieldTransformerConfig:
    """Config for testing field transformer."""
    def __init__(self):
        self.vocab_size = 1000
        self.field_hidden_size = 64
        self.hidden_size = 768
        self.ncols = 12
        self.num_layers = 2
        self.nhead = 8
        self.hidden_dropout_prob = 0.1
        self.pad_token_id = 0
        self.native_field_transformer = True


def test_native_field_block():
    """Test single field transformer block."""
    print("\n" + "="*60)
    print("NATIVE FIELD TRANSFORMER BLOCK TEST")
    print("="*60)

    d_model = 64
    nhead = 8
    dim_feedforward = 64
    batch_size = 2
    seq_len = 12  # number of fields

    # Create block
    block = NativeFieldTransformerBlock(d_model, nhead, dim_feedforward)
    print(f"✓ Created field transformer block")
    print(f"  d_model: {d_model}")
    print(f"  nhead: {nhead}")
    print(f"  dim_feedforward: {dim_feedforward}")

    # Test with (seq, batch, dim) format
    x = torch.randn(seq_len, batch_size, d_model)
    print(f"\nInput shape (seq, batch, dim): {x.shape}")

    try:
        with torch.no_grad():
            output = block(x)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {x.shape}")

        if output.shape == x.shape:
            print("✓ Forward pass successful (seq, batch, dim format)")
            seq_batch_ok = True
        else:
            print("✗ Output shape mismatch!")
            seq_batch_ok = False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        seq_batch_ok = False

    # Test with (batch, seq, dim) format
    x_batch_first = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape (batch, seq, dim): {x_batch_first.shape}")

    try:
        with torch.no_grad():
            output = block(x_batch_first)
        print(f"Output shape: {output.shape}")

        if output.shape == x_batch_first.shape:
            print("✓ Forward pass successful (batch, seq, dim format)")
            batch_seq_ok = True
        else:
            print("✗ Output shape mismatch!")
            batch_seq_ok = False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        batch_seq_ok = False

    return seq_batch_ok and batch_seq_ok


def test_native_field_encoder():
    """Test native field transformer encoder (stacked blocks)."""
    print("\n" + "="*60)
    print("NATIVE FIELD TRANSFORMER ENCODER TEST")
    print("="*60)

    d_model = 64
    nhead = 8
    num_layers = 2
    dim_feedforward = 64
    batch_size = 2
    seq_len = 12

    # Create encoder
    encoder = NativeFieldTransformerEncoder(d_model, nhead, num_layers, dim_feedforward)
    print(f"✓ Created field transformer encoder")
    print(f"  num_layers: {num_layers}")
    print(f"  d_model: {d_model}")

    # Test forward pass
    x = torch.randn(seq_len, batch_size, d_model)
    print(f"\nInput shape: {x.shape}")

    try:
        with torch.no_grad():
            output = encoder(x)
        print(f"Output shape: {output.shape}")

        if output.shape == x.shape:
            print("✓ Forward pass successful")
            return True
        else:
            print("✗ Output shape mismatch!")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rmsnorm_usage():
    """Verify that native field transformer uses RMSNorm."""
    print("\n" + "="*60)
    print("RMSNORM VERIFICATION")
    print("="*60)

    d_model = 64
    nhead = 8
    dim_feedforward = 64

    block = NativeFieldTransformerBlock(d_model, nhead, dim_feedforward)

    has_rmsnorm = False
    has_layernorm = False

    for name, module in block.named_modules():
        if isinstance(module, RMSNorm):
            has_rmsnorm = True
            print(f"✓ Found RMSNorm: {name}")
        elif isinstance(module, torch.nn.LayerNorm):
            has_layernorm = True
            print(f"✗ Found LayerNorm: {name}")

    print(f"\nSummary:")
    print(f"  RMSNorm present: {'YES ✓' if has_rmsnorm else 'NO ✗'}")
    print(f"  LayerNorm present: {'NO ✓' if not has_layernorm else 'YES ✗'}")

    return has_rmsnorm and not has_layernorm


def test_tabformer_embeddings_native():
    """Test TabFormerEmbeddings with native field transformer."""
    print("\n" + "="*60)
    print("TABFORMER EMBEDDINGS (NATIVE) TEST")
    print("="*60)

    config = FieldTransformerConfig()
    config.native_field_transformer = True

    # Create embeddings
    embeddings = TabFormerEmbeddings(config)
    print(f"✓ Created TabFormerEmbeddings (native)")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  field_hidden_size: {config.field_hidden_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  ncols: {config.ncols}")
    print(f"  num_layers: {config.num_layers}")

    # Verify uses native transformer
    is_native = embeddings.native_field_transformer
    print(f"  Using native field transformer: {'YES ✓' if is_native else 'NO ✗'}")

    # Test forward pass
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len, config.ncols))
    print(f"\nInput shape: {input_ids.shape} (batch, seq_len, ncols)")

    try:
        with torch.no_grad():
            output = embeddings(input_ids)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {config.hidden_size})")

        if output.shape == (batch_size, seq_len, config.hidden_size):
            print("✓ Forward pass successful")
            return True
        else:
            print("✗ Output shape mismatch!")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tabformer_embeddings_comparison():
    """Compare native vs original TabFormerEmbeddings."""
    print("\n" + "="*60)
    print("NATIVE VS ORIGINAL COMPARISON")
    print("="*60)

    config = FieldTransformerConfig()
    batch_size = 2
    seq_len = 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len, config.ncols))

    # Native version
    config.native_field_transformer = True
    embeddings_native = TabFormerEmbeddings(config)

    # Original version
    config.native_field_transformer = False
    embeddings_original = TabFormerEmbeddings(config)

    print(f"Input shape: {input_ids.shape}")

    try:
        with torch.no_grad():
            output_native = embeddings_native(input_ids)
            output_original = embeddings_original(input_ids)

        print(f"\nNative output shape: {output_native.shape}")
        print(f"Original output shape: {output_original.shape}")

        if output_native.shape == output_original.shape:
            print("✓ Output shapes match")
            return True
        else:
            print("✗ Output shapes don't match!")
            return False
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bidirectional_attention():
    """Verify that attention is bidirectional (not causal)."""
    print("\n" + "="*60)
    print("BIDIRECTIONAL ATTENTION TEST")
    print("="*60)

    d_model = 64
    nhead = 8
    dim_feedforward = 64

    block = NativeFieldTransformerBlock(d_model, nhead, dim_feedforward)

    # Check is_causal parameter in forward
    # We'll do a simple functional test: in bidirectional attention,
    # changing later positions should affect earlier positions in output

    seq_len = 10
    batch_size = 1
    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        # Original output
        out1 = block(x.clone())

        # Modify last position
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(batch_size, d_model) * 10

        # New output
        out2 = block(x_modified)

        # In bidirectional attention, modifying position N should affect position 0
        # In causal attention, it wouldn't
        diff_at_first_pos = (out1[:, 0, :] - out2[:, 0, :]).abs().mean().item()

    print(f"Difference at first position when last position changed: {diff_at_first_pos:.6f}")

    if diff_at_first_pos > 1e-5:
        print("✓ Bidirectional attention confirmed (first position affected by last)")
        return True
    else:
        print("✗ Might be causal attention (first position not affected by last)")
        return False


def main():
    print("="*60)
    print("NATIVE FIELD TRANSFORMER TEST SUITE")
    print("="*60)

    # Run all tests
    block_ok = test_native_field_block()
    encoder_ok = test_native_field_encoder()
    rmsnorm_ok = test_rmsnorm_usage()
    embeddings_ok = test_tabformer_embeddings_native()
    comparison_ok = test_tabformer_embeddings_comparison()
    bidirectional_ok = test_bidirectional_attention()

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Field transformer block: {'PASS ✓' if block_ok else 'FAIL ✗'}")
    print(f"Field transformer encoder: {'PASS ✓' if encoder_ok else 'FAIL ✗'}")
    print(f"RMSNorm usage: {'PASS ✓' if rmsnorm_ok else 'FAIL ✗'}")
    print(f"TabFormerEmbeddings (native): {'PASS ✓' if embeddings_ok else 'FAIL ✗'}")
    print(f"Native vs Original comparison: {'PASS ✓' if comparison_ok else 'FAIL ✗'}")
    print(f"Bidirectional attention: {'PASS ✓' if bidirectional_ok else 'FAIL ✗'}")

    all_pass = all([block_ok, encoder_ok, rmsnorm_ok, embeddings_ok, comparison_ok, bidirectional_ok])
    print(f"\nOverall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("="*60)


if __name__ == "__main__":
    main()
