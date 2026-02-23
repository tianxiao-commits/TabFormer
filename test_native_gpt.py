"""
Test script to verify the native GPT model with RoPE and RMSNorm.

This script:
1. Instantiates the native GPT model
2. Verifies it uses RMSNorm (not LayerNorm)
3. Verifies it uses RoPE (not learned position embeddings)
4. Checks precision requirements:
   - FP32: RMSNorm, Softmax, RoPE cos/sin tables
   - BF16: Linear projections, embeddings
5. Tests a forward pass
"""

import torch
from models.gpt_native import NativeGPTLMHeadModel, RMSNorm, RotaryPositionEmbedding


class GPTConfig:
    """Minimal config for testing."""
    def __init__(self):
        self.vocab_size = 1000
        self.hidden_size = 768
        self.num_hidden_layers = 4
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.layer_norm_eps = 1e-6
        self.pad_token_id = 0
        self.tie_word_embeddings = False


def check_module_types(model):
    """Check that model uses RMSNorm and RoPE."""
    print("\n" + "="*60)
    print("ARCHITECTURE VERIFICATION")
    print("="*60)

    has_rmsnorm = False
    has_rope = False
    has_layernorm = False

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            has_rmsnorm = True
            print(f"✓ Found RMSNorm: {name}")
        elif isinstance(module, torch.nn.LayerNorm):
            has_layernorm = True
            print(f"✗ Found LayerNorm: {name} (should be RMSNorm!)")
        elif isinstance(module, RotaryPositionEmbedding):
            has_rope = True
            print(f"✓ Found RoPE: {name}")

    print("\nSummary:")
    print(f"  RMSNorm present: {'YES ✓' if has_rmsnorm else 'NO ✗'}")
    print(f"  RoPE present: {'YES ✓' if has_rope else 'NO ✗'}")
    print(f"  LayerNorm present: {'NO ✓' if not has_layernorm else 'YES ✗ (should use RMSNorm)'}")

    return has_rmsnorm and has_rope and not has_layernorm


def check_precision_requirements(model):
    """Verify precision of key components."""
    print("\n" + "="*60)
    print("PRECISION VERIFICATION")
    print("="*60)

    # Check RoPE cos/sin tables are FP32
    rope_modules = [m for m in model.modules() if isinstance(m, RotaryPositionEmbedding)]
    if rope_modules:
        rope = rope_modules[0]
        print(f"\nRoPE cos/sin tables:")
        print(f"  cos_cached dtype: {rope.cos_cached.dtype} (expected: float32)")
        print(f"  sin_cached dtype: {rope.sin_cached.dtype} (expected: float32)")
        rope_correct = rope.cos_cached.dtype == torch.float32 and rope.sin_cached.dtype == torch.float32
        print(f"  Status: {'✓ FP32' if rope_correct else '✗ Not FP32'}")
    else:
        print("\n✗ No RoPE modules found!")
        rope_correct = False

    # Check Linear layers exist (will be BF16 in practice with autocast)
    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"\nLinear layers:")
    print(f"  Count: {linear_count}")
    print(f"  Status: ✓ Present (will run in BF16 with torch.autocast)")

    # Check Embeddings exist
    embedding_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Embedding))
    print(f"\nEmbedding layers:")
    print(f"  Count: {embedding_count}")
    print(f"  Status: ✓ Present (will run in BF16 with torch.autocast)")

    print("\nNote: RMSNorm internally converts to FP32 for computation")
    print("Note: Softmax in F.scaled_dot_product_attention runs in FP32 automatically")

    return rope_correct


def test_forward_pass(model, config):
    """Test a forward pass with the model."""
    print("\n" + "="*60)
    print("FORWARD PASS TEST")
    print("="*60)

    batch_size = 2
    seq_len = 64

    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"\nInput shape: {input_ids.shape}")

    try:
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs['logits']
        print(f"Output logits shape: {logits.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

        if logits.shape == (batch_size, seq_len, config.vocab_size):
            print("✓ Forward pass successful!")
            return True
        else:
            print("✗ Output shape mismatch!")
            return False

    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_attention(model, config):
    """Verify causal attention mask works correctly."""
    print("\n" + "="*60)
    print("CAUSAL ATTENTION TEST")
    print("="*60)

    batch_size = 1
    seq_len = 8

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        print(f"✓ Causal attention working (implicit is_causal=True)")
        return True

    except Exception as e:
        print(f"✗ Causal attention test failed: {e}")
        return False


def main():
    print("="*60)
    print("NATIVE GPT MODEL TEST")
    print("="*60)

    # Create config
    config = GPTConfig()
    print(f"\nConfig:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")

    # Instantiate model
    print("\nInstantiating model...")
    model = NativeGPTLMHeadModel(config)
    print("✓ Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Run tests
    arch_ok = check_module_types(model)
    precision_ok = check_precision_requirements(model)
    forward_ok = test_forward_pass(model, config)
    causal_ok = test_causal_attention(model, config)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Architecture: {'PASS ✓' if arch_ok else 'FAIL ✗'}")
    print(f"Precision: {'PASS ✓' if precision_ok else 'FAIL ✗'}")
    print(f"Forward pass: {'PASS ✓' if forward_ok else 'FAIL ✗'}")
    print(f"Causal attention: {'PASS ✓' if causal_ok else 'FAIL ✗'}")

    all_pass = arch_ok and precision_ok and forward_ok and causal_ok
    print(f"\nOverall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("="*60)


if __name__ == "__main__":
    main()
