from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGS_DIR = EXPERIMENTS_DIR / "figs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_NAME = "glue"            # <-- required
DATASET_CONFIG = "sst2"          # <-- required
TEXT_FIELD = "sentence"          # <-- required
LABEL_FIELD = "label"            # <-- required
MAX_SAMPLES = None               # or small number for debugging

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
PIPELINE_TASK = "text-classification"
DEVICE = -1   # CPU

# Noise settings
NOISE_TYPES = [
    "emoji",
    "spelling",
    "slang",
    "hashtag",
    "repetition",
    "codeswitch",
]

NOISE_INTENSITIES = [0.1, 0.3, 0.5]

RANDOM_SEED = 42
