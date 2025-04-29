import os
import yaml

def load_config(config_path='config.yml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model_config(model_type):
    """Get configuration for a specific model type."""
    config = load_config()
    return config['models'].get(model_type, {})