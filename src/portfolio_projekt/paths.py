from pathlib import Path


# Repository root:
# .../Portfolio Projekt/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data
DATA_DIR = PROJECT_ROOT / "data"

DATA_UCI_RAW = DATA_DIR / "uci" / "raw"
DATA_UCI_INTERIM = DATA_DIR / "uci" / "interim"
DATA_UCI_PROCESSED = DATA_DIR / "uci" / "processed"

DATA_DL3_RAW = DATA_DIR / "dl3" / "raw"

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_ML_DIR = RESULTS_DIR / "ml"
RESULTS_ML_FIGURES = RESULTS_ML_DIR / "figures"
RESULTS_ML_TABLES = RESULTS_ML_DIR / "tables"

RESULTS_DL3_DIR = RESULTS_DIR / "dl3"
RESULTS_DL3_FIGURES = RESULTS_DL3_DIR / "figures"
RESULTS_DL3_TABLES = RESULTS_DL3_DIR / "tables"