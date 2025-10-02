from pathlib import Path

MA_PERIODS = [5, 20, 60, 120]
TARGETS = ["usd", "cny", "jpy", "eur", "gbp"]
BASE_FEATURES = ["dxy", "wti", "vix", "dgs10", "kr_rate", "us_rate", "kr_us_diff"]
LOOK_BACK = 60

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PRED_CSV_PTH = BASE_DIR / "2024_predictions.csv"
