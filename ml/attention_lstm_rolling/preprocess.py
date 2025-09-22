import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from constant import FEATURES, LOOK_BACK


def add_moving_average_features(
    df: pd.DataFrame, target: str, alphas=None
) -> pd.DataFrame:
    if alphas is None:
        alphas = {5: 0.1, 20: 0.1, 60: 0.1, 120: 0.1}

    for period in alphas.keys():
        df[f"{target}_MA{period}"] = df[target].rolling(period).mean()
        df[f"{target}_EMA{period}"] = (
            df[target].ewm(alpha=alphas[period], adjust=False).mean()
        )
    return df


def create_sequences(features, targets, look_back):
    X = [features[i : i + look_back] for i in range(len(features) - look_back)]
    y = [targets[i + look_back] for i in range(len(features) - look_back)]
    return np.array(X), np.array(y)


def filter_dataframe(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df


def load_preprocess_data(target: str, alphas=None):
    from ml.data_merge import create_merged_dataset

    if alphas is None:
        alphas = {5: 0.1, 20: 0.1, 60: 0.1, 120: 0.1}
    df = create_merged_dataset()
    if df is None or df.empty:
        raise ValueError("데이터 로딩 실패")
    df["kr_us_diff"] = df["kr_rate"] - df["us_rate"]
    df = add_moving_average_features(df, target, alphas)
    ma_ema_periods = [5, 20, 60, 120]
    ma_ema_features = [
        f"{target}_{kind}{p}" for kind in ["MA", "EMA"] for p in ma_ema_periods
    ]
    features = list(dict.fromkeys(FEATURES + ma_ema_features))
    df = df.dropna(subset=features + [target])
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(df[features])
    target_scaled = target_scaler.fit_transform(df[[target]])
    X_seq, y_seq = create_sequences(features_scaled, target_scaled, LOOK_BACK)
    return {
        "df": df,
        "features": features,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "X_seq": X_seq,
        "y_seq": y_seq,
    }
