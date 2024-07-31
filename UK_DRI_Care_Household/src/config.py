from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# setting the root directory path
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "data.db"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

PREPROC_FILE = PROJ_ROOT / "src/data_raw_prep.py"
MODELS_DIR = PROJ_ROOT / "src/models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_FOR_MODEL_PATH = PROCESSED_DATA_DIR / 'data_for_model.pkl'

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass