"""
Configuration file containing paths, hyperparameters, and device setup.
"""
from pathlib import Path
import torch

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"

# Ensure essential output directories exist
for d in (RAW_DIR, CACHE_DIR, CHECKPOINT_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Spatiotemporal sequence parameters
SEQ_LENGTH = 12        # 12 steps of 5 minutes = 1 hour lookback horizon
NUM_SENSORS = 207       # METR-LA total loop detector sensors
TRAIN_RATIO = 0.70      # 70% Chronological train split
VAL_RATIO = 0.15        # 15% Validation split
TEST_RATIO = 0.15       # 15% Test split

# Compute device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
