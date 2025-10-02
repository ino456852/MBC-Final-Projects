import sys
from pathlib import Path
from tensorflow import keras

# 현재 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from .data_processor import DataProcessor
from model import ResidualBlock
from .constant import LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE


def load_model(target: str):
    model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}"
    return keras.models.load_model(
        model_path, compile=False, custom_objects={"ResidualBlock": ResidualBlock}
    )


def predict_next_day():
    data_processor = DataProcessor()
    targets = data_processor.targets
    pred_values = {}

    for target in targets:
        data = data_processor.get_proceed_data(target=target)

        model = load_model(target)

        X = data.drop(columns=[target])
        X_last = X.iloc[-LOOK_BACK:]

        feature_scaler = data_processor.get_feature_scaler(target=target)
        X_input_scaled = feature_scaler.transform(X_last).reshape(1, LOOK_BACK, -1)

        y_pred_scaled = model.predict(X_input_scaled)

        target_scaler = data_processor.get_target_scaler(target=target)
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()[0]

        pred_values[target] = y_pred

    return pred_values


if __name__ == "__main__":
    print(predict_next_day())
