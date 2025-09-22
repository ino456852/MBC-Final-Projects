import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Optional, Tuple
from constant import TARGETS, FEATURES, LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE
from preprocess import (
    add_moving_average_features,
    create_sequences,
    filter_dataframe,
    load_preprocess_data,
)
from model import Attention


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def prepare_features_and_scalers(df: pd.DataFrame, target: str):
    df_target = add_moving_average_features(df, target)
    ma_ema_periods = [5, 20, 60, 120]
    ma_ema_features = [
        f"{target}_{kind}{p}" for kind in ["MA", "EMA"] for p in ma_ema_periods
    ]
    features = list(dict.fromkeys(FEATURES + ma_ema_features))
    df_target = df_target.dropna(subset=features + [target])
    return (
        features,
        MinMaxScaler().fit(df_target[features]),
        MinMaxScaler().fit(df_target[[target]]),
    )


def load_models_and_scalers():
    # 데이터프레임은 load_preprocess_data에서 가져오는 것이 더 일관적입니다.
    df = load_preprocess_data(TARGETS[0])["df"]
    models, scalers, features_map = {}, {}, {}
    for target in TARGETS:
        model_path = os.path.join(MODEL_DIR, f"{target}{KERAS_FILE_TEMPLATE}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_path}가 없습니다.")
        models[target] = tf.keras.models.load_model(
            model_path, custom_objects={"Attention": Attention}
        )
        features, f_scaler, t_scaler = prepare_features_and_scalers(df, target)
        features_map[target] = features
        scalers[target] = {"feature": f_scaler, "target": t_scaler}
    return df, models, scalers, features_map


def get_predictions(
    df: pd.DataFrame,
    models: dict,
    scalers: dict,
    features_map: dict,
    target: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    future_steps: int = 0,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[pd.DatetimeIndex],
    Optional[dict],
]:
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")

    df = add_moving_average_features(df, target).dropna(
        subset=features_map[target] + [target]
    )
    df = filter_dataframe(df, start_date, end_date)

    if future_steps > 0:
        if len(df) < LOOK_BACK:
            return None, None, None, None
        input_data = scalers[target]["feature"].transform(
            df[features_map[target]].iloc[-LOOK_BACK:]
        )
        input_seq = np.expand_dims(input_data, axis=0)
        pred_scaled = models[target].predict(input_seq)
        y_pred = scalers[target]["target"].inverse_transform(pred_scaled).flatten()
        future_date = df.index[-1] + pd.Timedelta(days=future_steps)
        return None, y_pred, pd.DatetimeIndex([future_date]), None

    if len(df) <= LOOK_BACK:
        return None, None, None, None

    X_scaled = scalers[target]["feature"].transform(df[features_map[target]])
    y_scaled = scalers[target]["target"].transform(df[[target]])
    X, y_true = create_sequences(X_scaled, y_scaled, LOOK_BACK)
    y_pred = models[target].predict(X)

    y_true_inv = scalers[target]["target"].inverse_transform(y_true)
    y_pred_inv = scalers[target]["target"].inverse_transform(y_pred)
    dates = df.index[LOOK_BACK:]

    metrics = evaluate_predictions(y_true_inv, y_pred_inv)
    return y_true_inv, y_pred_inv, dates, metrics
