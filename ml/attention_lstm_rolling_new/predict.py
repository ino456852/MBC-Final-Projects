from .data_processor import DataProcessor
from .model import Attention
from .constant import LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE
from tensorflow import keras


def load_model(target: str):
    model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}"
    return keras.models.load_model(
        model_path, compile=False, custom_objects={"Attention": Attention}
    )


def predict_next_day():
    data_processor = DataProcessor()
    targets = data_processor.targets
    pred_values = {}

    for target in targets:
        data = data_processor.get_proceed_data(target=target)

        model = load_model(target)

        # feature 컬럼만 추출 (target 컬럼 제외)
        X = data.drop(columns=[target])
        X_last = X.iloc[-LOOK_BACK:]

        # feature scaling
        feature_scaler = data_processor.get_feature_scaler(target=target)
        X_input_scaled = feature_scaler.transform(X_last).reshape(1, LOOK_BACK, -1)

        # 예측
        y_pred_scaled = model.predict(X_input_scaled)

        # 역변환
        target_scaler = data_processor.get_target_scaler(target=target)
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()[0]

        pred_values[target] = y_pred

    return pred_values


if __name__ == "__main__":
    print(predict_next_day())
