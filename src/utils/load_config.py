import yaml

def load_config(config_file):
    try:
        with open(config_file, "r") as f:
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