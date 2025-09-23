import copy
from pathlib import Path

CONFIG = {
    # ------------------------------
    # 모델 구조 관련 파라미터 (튜닝의 기본값/탐색 범위로 사용됨)
    # ------------------------------
    # LSTM 레이어의 뉴런(유닛) 개수
    "LSTM_UNITS": 150,
    # Bidirectional LSTM의 두 번째 LSTM 레이어 유닛 개수 (Cross-Attention 모델용)
    "LSTM_UNITS_HALF": 75,
    # Dropout 비율 (과적합 방지)
    "DROPOUT_RATE": 0.3,
    # Multi-Head Attention의 헤드 개수
    "NUM_HEADS": 8,
    # Multi-Head Attention의 Key Dimension
    "KEY_DIM": 64,
    # ------------------------------
    # 훈련 과정 관련 파라미터
    # ------------------------------
    # 총 훈련 횟수 (Epochs)
    "EPOCHS": 100,
    # 한 번에 훈련에 사용할 데이터 샘플의 수 (Batch Size)
    "BATCH_SIZE": 32,
    # 훈련 데이터 중 검증 데이터로 사용할 비율
    "VALIDATION_SPLIT": 0.1,
    # Early Stopping 조건: 검증 손실이 n번 이상 개선되지 않으면 훈련 중단
    "PATIENCE": 10,
    # 옵티마이저(Adam)의 학습률
    "LEARNING_RATE": 0.001,
}


def get_model_config():
    """CONFIG 딕셔너리의 복사본을 반환"""
    return copy.deepcopy(CONFIG)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"  # 현재폴더/models
BEST_PARAMS_PTH = BASE_DIR / "best_params.json"  # 현재폴더/best_params.json


TARGETS = ["usd", "cny", "jpy", "eur", "gbp"]
BASE_FEATURES = ["dxy", "wti", "vix", "dgs10", "kr_rate", "us_rate", "kr_us_diff"]

# ------------------------------
# 데이터 관련 파라미터
# ------------------------------
# 과거 몇일의 데이터를 보고 다음날을 예측할지 결정
LOOK_BACK = 60
# 사용할 이동평균 및 지수이동평균 기간(일) 리스트
MA_PERIODS = [5, 20, 60, 120]
# TimeSeriesSplit을 위한 분할(fold) 개수. (n-1)/n 비율로 훈련/테스트 데이터가 나뉨.
# 예: 5 -> 4/5는 훈련(80%), 1/5는 테스트(20%)
N_SPLITS = 5
