"""
Model configuration presets for TabFormer benchmarking.
Provides 120M and 20M parameter model configurations.
"""

MODEL_CONFIGS = {
    '120M': {
        'description': '120M parameter model with 16 layers',
        'field_hidden_size': 64,
        'num_hidden_layers': 16,
        'hidden_size': 768,
        'intermediate_size': 3072,
        'num_attention_heads': 12,
        'ncols': 12,
    },
    '20M': {
        'description': '20M parameter model with 4 layers',
        'field_hidden_size': 48,
        'num_hidden_layers': 4,
        'hidden_size': 576,  # 48 * 12
        'intermediate_size': 1536,
        'num_attention_heads': 12,
        'ncols': 12,
    }
}

def get_model_config(config_name):
    """
    Get model configuration by name.

    Args:
        config_name: Name of the config ('120M' or '20M')

    Returns:
        Dictionary with model configuration parameters
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[config_name]

def estimate_parameters(config):
    """
    Estimate number of parameters for a given config.
    Rough approximation for BERT-like models.

    Args:
        config: Model configuration dictionary

    Returns:
        Estimated number of parameters in millions
    """
    vocab_size = 30522
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_layers = config['num_hidden_layers']

    # Embedding layer
    embeddings = vocab_size * config['field_hidden_size']

    # Each transformer layer has:
    # - Attention: 4 * hidden_size^2 (Q, K, V, output projections)
    # - FFN: 2 * hidden_size * intermediate_size
    # - Layer norms: ~4 * hidden_size (small, ignore)
    per_layer = 4 * (hidden_size ** 2) + 2 * hidden_size * intermediate_size
    transformer_layers = num_layers * per_layer

    # Output projection
    output_layer = hidden_size * vocab_size

    total_params = embeddings + transformer_layers + output_layer
    return total_params / 1e6  # Convert to millions

if __name__ == '__main__':
    print("TabFormer Model Configurations:")
    print("=" * 60)
    for name, config in MODEL_CONFIGS.items():
        params = estimate_parameters(config)
        print(f"\n{name} ({config['description']}):")
        print(f"  Estimated parameters: {params:.1f}M")
        print(f"  Layers: {config['num_hidden_layers']}")
        print(f"  Hidden size: {config['hidden_size']}")
        print(f"  Field hidden size: {config['field_hidden_size']}")
        print(f"  Intermediate size: {config['intermediate_size']}")
