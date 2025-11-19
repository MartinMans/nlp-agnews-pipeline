from pathlib import Path

# Random seed used across scripts
RANDOM_STATE = 1

# Data directories
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

# Label mapping for AG News
ID2LABEL = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}
