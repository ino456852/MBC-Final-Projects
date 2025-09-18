import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import sys, os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ml_path = os.path.join(base_path, 'ml')
if base_path not in sys.path: sys.path.insert(0, base_path)
if ml_path not in sys.path: sys.path.insert(0, ml_path)

from ml.data_merge import create_merged_dataset
from model_config import LOOK_BACK, N_SPLITS, MA_WINDOWS

TARGETS = ['usd', 'cny', 'jpy', 'eur', 'gbp']
BASE_FEATURES = ['dxy', 'wti', 'vix', 'dgs10', 'kr_rate', 'us_rate', 'kr_us_diff']

def get_master_dataframe() -> pd.DataFrame:
    master_df = create_merged_dataset()
    if master_df is None or master_df.empty:
        raise ValueError("데이터를 가져오지 못했거나 데이터가 비어있습니다.")
    master_df['kr_us_diff'] = master_df['kr_rate'] - master_df['us_rate']
    required_cols = TARGETS + list(set(BASE_FEATURES) - set(['kr_us_diff']))
    missing = [c for c in required_cols if c not in master_df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 부족: {missing}")
    return master_df

def create_sequences(features, target, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i:(i + look_back)])
        y.append(target[i + look_back])
    return np.array(X), np.array(y)

def prepare_data_for_target(master_df: pd.DataFrame, target_name: str):
    model_data = master_df.copy()
    ma_features = []
    for window in MA_WINDOWS:
        ma_col = f'{target_name}_MA{window}'
        model_data[ma_col] = model_data[target_name].rolling(window).mean()
        ma_features.append(ma_col)
        ema_col = f'{target_name}_EMA{window}'
        model_data[ema_col] = model_data[target_name].ewm(span=window, adjust=False).mean()
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
    
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled)
    X_test, _ = create_sequences(test_features_scaled, np.zeros(len(test_features_scaled)))

    return X_train, y_train, X_test, test_df.index[LOOK_BACK:], feature_scaler, target_scaler