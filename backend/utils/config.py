"""Configuration module."""

from pathlib import Path

# Directory paths
# Project root: .../nasa-app
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model artifacts live in project_root/data-modeling/artifacts
SERIALIZED_DIR = PROJECT_ROOT / "data-modeling" / "artifacts"
SERIALIZED_DIR.mkdir(parents=True, exist_ok=True)
