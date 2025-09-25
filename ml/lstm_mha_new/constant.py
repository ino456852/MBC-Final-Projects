from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "scalers"
BEST_PARAMS_PTH = BASE_DIR / "best_params.json"
LOOK_BACK = 60
PRED_TRUE_DIR = BASE_DIR / "pred_true"
PRED_TRUE_CSV = "pred_true.csv"
