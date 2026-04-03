import yaml
from pathlib import Path

# Project root directory (image_classification/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

def load_config(config_file):
    try:
        config_path = CONFIGS_DIR / config_file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def save_config(config, config_file):
    try:
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
    except Exception as e:
        print(f"Error saving config file: {e}")