"""Configuration module."""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INSIGHTS_DIR = BASE_DIR / "insights"
CORRELATION_DIR = INSIGHTS_DIR / "correlation"
CORRELATION_DIR.mkdir(parents=True, exist_ok=True)
SERIALIZED_DIR.mkdir(parents=True, exist_ok=True)
PERFORMANCE_DIR = INSIGHTS_DIR / "performance"
PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
SERIALIZED_DIR = INSIGHTS_DIR / "serialized_artifacts"
SERIALIZED_DIR.mkdir(parents=True, exist_ok=True)
