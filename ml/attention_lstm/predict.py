import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from .constatnt import LOOK_BACK, MODEL_DIR, TARGETS
from .preprocess import get_procceed_data
from .model import CustomAttention


def load_model_and_scaler(target: str):
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, f"model_{target}.keras"),
        custom_objects={"CustomAttention": CustomAttention},  # custom_objects 추가
    )
    with open(os.path.join(MODEL_DIR, f"scaler_{target}.pkl"), "rb") as f:
        scalers = pickle.load(f)
    return model, scalers


def predict_future(target: str, data: pd.DataFrame):
    """서비스용 미래 예측 함수"""
    model, scalers = load_model_and_scaler(target)
    feature_scaler = scalers["feature_scaler"]
    target_scaler = scalers["target_scaler"]
    features = scalers["features"]

    X_recent = feature_scaler.transform(data[features])
    current_sequence = X_recent[-LOOK_BACK:]

    pred_scaled = model.predict(
        current_sequence.reshape(1, LOOK_BACK, len(features)), verbose=0
    )
    pred_value = target_scaler.inverse_transform(pred_scaled)[0, 0]

    current_sequence = np.roll(current_sequence, -1, axis=0)
    current_sequence[-1] = X_recent[-1]

    return pred_value


def predict_next_day() -> dict:
    data=get_procceed_data()
    pred_data = {}

    for target in TARGETS:
        pred_data[target] = predict_future(target, data)

    return pred_data


if __name__ == "__main__":
    print(predict_next_day())
