import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from .preprocessing import LOOK_BACK, BASE_FEATURES, TARGETS


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def load_model_and_scaler(target: str):
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f'model_{target}.keras'))
    with open(os.path.join(MODEL_DIR, f'scaler_{target}.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    return model, scalers


def predict_future(target: str, recent_data: pd.DataFrame, days: int = 1):
    """서비스용 미래 예측 함수"""
    model, scalers = load_model_and_scaler(target)
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']
    features = scalers['features']

    X_recent = feature_scaler.transform(recent_data[features])
    current_sequence = X_recent[-LOOK_BACK:]

    predictions = []
    for _ in range(days):
        pred_scaled = model.predict(current_sequence.reshape(1, LOOK_BACK, len(features)), verbose=0)
        pred_value = target_scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred_value)

        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = X_recent[-1]

    return predictions
