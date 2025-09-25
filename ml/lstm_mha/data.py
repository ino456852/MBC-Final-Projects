import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from ..data_merge import create_merged_dataset
from .constant import LOOK_BACK, N_SPLITS, TARGETS, BASE_FEATURES, MA_PERIODS


def get_cached_data():
    master_df = create_merged_dataset()
    master_df["kr_us_diff"] = master_df["kr_rate"] - master_df["us_rate"]
    required_cols = TARGETS + [col for col in BASE_FEATURES if col != "kr_us_diff"]
    missing = [c for c in required_cols if c not in master_df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 부족: {missing}")
    yield master_df


def get_master_dataframe():
    """캐싱된 master_df를 제너레이터로 반환"""
    gen = get_cached_data()
    return next(gen).copy()


def create_sequences(features, target, look_back):
    X, y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i : (i + look_back)])
        y.append(target[i + look_back])
    return np.array(X), np.array(y)


def prepare_data_for_target(df: pd.DataFrame, target_name: str):
    model_data = df.copy()
    ma_features = []
    for period in MA_PERIODS:
        ma_col = f"{target_name}_MA{period}"
        model_data[ma_col] = model_data[target_name].rolling(period).mean()
        ma_features.append(ma_col)
        ema_col = f"{target_name}_EMA{period}"
        model_data[ema_col] = (
            model_data[target_name].ewm(span=period, adjust=False).mean()
        )
        ma_features.append(ema_col)
    model_data.dropna(inplace=True)

    features = BASE_FEATURES.copy()
    other_currencies = [c for c in TARGETS if c != target_name]
    features.extend(other_currencies)
    features.extend(ma_features)

    print(f"[{target_name.upper()} 예측] 사용된 피처 개수: {len(features)}")
    data_subset = model_data[features + [target_name]]

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    all_splits = list(tss.split(data_subset))
    train_indices, test_indices = all_splits[-1]
    train_df = data_subset.iloc[train_indices]
    test_df = data_subset.iloc[test_indices]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features_scaled = feature_scaler.fit_transform(train_df[features])
    train_target_scaled = target_scaler.fit_transform(train_df[[target_name]])
    test_features_scaled = feature_scaler.transform(test_df[features])

    X_train, y_train = create_sequences(
        features=train_features_scaled, target=train_target_scaled, look_back=LOOK_BACK
    )
    X_test, _ = create_sequences(
        features=test_features_scaled,
        target=np.zeros(len(test_features_scaled)),
        look_back=LOOK_BACK,
    )

    return (
        X_train,
        y_train,
        X_test,
        test_df.index[LOOK_BACK:],
        feature_scaler,
        target_scaler,
    )
