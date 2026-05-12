import os
import yaml
from pathlib import Path

# Project root directory (image_classification/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

def _candidate_config_paths(config_file: str) -> list[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path))

    path = Path(config_file)
    if path.is_absolute():
        candidates.append(path)

    candidates.extend([
        CONFIGS_DIR / config_file,
        PROJECT_ROOT / config_file,
        Path.cwd() / config_file,
        Path("/app/configs") / config_file,
    ])

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates

def load_config(config_file: str):
    tried_paths = []
    for config_path in _candidate_config_paths(config_file):
        tried_paths.append(str(config_path))
        if config_path.is_file():
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                return None

    print("Error loading config file: not found")
    print("Tried paths:", ", ".join(tried_paths))
    return None

def save_config(config, config_file):
    try:
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
    except Exception as e:
        print(f"Error saving config file: {e}")