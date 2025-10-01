import pandas as pd
import numpy as np
import joblib, os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from ..data_merge import create_merged_dataset
from .constant import (PRED_TRUE_DIR, LOOK_BACK, PRED_TRUE_CSV, SCALER_DIR)

class DataProcessor:
    def __init__(self):
        self.origin_data = create_merged_dataset()
        self.origin_data["kr_us_diff"] = (self.origin_data["kr_rate"] - self.origin_data["us_rate"])
        self.origin_data["us_jp_diff"] = (self.origin_data["dgs10"] - self.origin_data["jpy10"])
        self.origin_data["us_eu_diff"] = (self.origin_data["dgs10"] - self.origin_data["eur10"])
        
        self.targets = ["usd", "cny", "jpy", "eur"]
        self.feature_sets = {
            "usd": ["vix","wti","dgs10","cny"],
            "cny": ["vix", "wti", "dxy", "usd", "dgs10"],
            "jpy": ["us_jp_diff","vix","dxy", "dgs10", "jpy10"], 
            "eur": ["us_eu_diff", "eur10", "dxy", "vix"]
        }
        
        # 변수별 스케일러 매핑 정의
        self.var_scaler_map = {
            "robust": ["vix", "wti", "dxy", "usd", "cny"],
            "standard": ["us_jp_diff", "us_eu_diff", "dgs10", "jpy10", "eur10"]
        }

    def add_indicators(self, data: pd.DataFrame, target: str):
        periods = [5, 20, 60, 120]
        for p in periods:
            data[f"MA_{p}"] = data[target].rolling(p).mean()
            data[f"EMA_{p}"] = data[target].ewm(span=p, adjust=False).mean()
        data.dropna(inplace=True)

    def get_target_scaler(self, target: str):
        scaler_path = SCALER_DIR / f"{target}_target_scaler.pkl"
        return joblib.load(scaler_path)

    def get_feature_scaler(self, target: str):
        file_path = os.path.join(SCALER_DIR, f"{target}_feature_scalers.pkl") 
        if not os.path.exists(file_path):
            return None
        return joblib.load(file_path)

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray, seq_len: int, y_index=None
    ):
        Xs, ys, idxs = [], [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
            if y_index is not None:
                idxs.append(y_index[i + seq_len])
        if y_index is not None:
            return np.array(Xs), np.array(ys), np.array(idxs)
        return np.array(Xs), np.array(ys)

    # 실제값, 예측값 csv 저장 (역변환)
    def save_predictions_csv(self, y_true, y_pred, target: str, index):
        scaler = self.get_target_scaler(target=target)
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        df = pd.DataFrame({"true": y_true_inv, "pred": y_pred_inv})
        df.index = index

        csv_file = f"{target}_{PRED_TRUE_CSV}"
        os.makedirs(PRED_TRUE_DIR, exist_ok=True)
        df.to_csv(PRED_TRUE_DIR / csv_file)
        print(f"Saved: {csv_file}")

    # targets에 없는 컬럼 + 타겟 컬럼만 남기고 나머지 제거
    def get_proceed_data(self, target) -> pd.DataFrame:
        data = self.origin_data.copy()
        
        feature_cols = self.feature_sets.get(target, [])
        keep_cols = feature_cols + [target]
        
        missing_cols = [col for col in keep_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataProcessor: {missing_cols}")        
        data = data[keep_cols]
        
        self.add_indicators(data=data, target=target)
        return data

    def get_scaler(self, scaler_type="standard"):
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "robust":
            return RobustScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def get_sequence_data(self, target: str):
        data = self.get_proceed_data(target=target)

        X = data.drop(columns=[target])
        y = data[target]

        target_scaler = self.get_scaler(scaler_type="minmax")
        target_scaled = target_scaler.fit_transform(y.to_frame())
        
        features_scaled_list = []
        feature_scalers = {}
        current_features = self.feature_sets.get(target, [])
        
        # 특징 컬럼들을 반복하면서 스케일러 적용
        for var in current_features:
            scaler_type = None
            if var in self.var_scaler_map.get("robust", []):
                scaler_type = "robust"
            elif var in self.var_scaler_map.get("standard", []):
                scaler_type = "standard"
            else:
                scaler_type = "standard"
            var_data = X[[var]]
            
            # 스케일러 생성 및 적용
            scaler = self.get_scaler(scaler_type=scaler_type)
            var_scaled = scaler.fit_transform(var_data)
            
            # 스케일러 저장 (역변환을 위해 필요)
            feature_scalers[var] = scaler
            features_scaled_list.append(var_scaled)

        # 스케일링된 특징들을 다시 하나의 배열로 합치기
        features_scaled = np.hstack(features_scaled_list)
        
        # 스케일러 파일 저장
        os.makedirs(SCALER_DIR, exist_ok=True)
        joblib.dump(feature_scalers, os.path.join(SCALER_DIR, f"{target}_feature_scalers.pkl"))

        # 기존 타겟 스케일러 저장 로직
        joblib.dump(target_scaler, os.path.join(SCALER_DIR, f"{target}_target_scaler.pkl"))
        
        # 시퀀스 데이터 생성
        X_seq, y_seq, y_idxs = self.create_sequences(
            features_scaled,
            target_scaled,
            seq_len=LOOK_BACK,
            y_index=np.arange(len(data))
        )        
        return X_seq, y_seq, y_idxs