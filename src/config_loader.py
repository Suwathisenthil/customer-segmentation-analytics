"""config_loader.py - Load and expose project configuration."""
import yaml
from pathlib import Path


def load_config(path: str = "config/config.yaml") -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)


CFG = load_config()
