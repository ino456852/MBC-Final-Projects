import os
from pathlib import Path

TARGETS = ["usd", "eur", "gbp", "cny", "jpy"]
FEATURES = ["dxy", "wti", "vix", "dgs10", "kr_rate", "us_rate", "kr_us_diff"]

LOOK_BACK = 60

MODEL_DIR = Path(__file__).resolve().parent / "models"
KERAS_FILE_TEMPLATE = "_attention_lstm.keras"

os.makedirs(MODEL_DIR, exist_ok=True)
