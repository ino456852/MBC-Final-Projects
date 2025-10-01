import numpy as np
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
        
        feature_scalers_dict = data_processor.get_feature_scaler(target=target)
        
        if feature_scalers_dict is None:
            print(f"Error: Feature scalers not found for {target}")
            continue

        scaled_features = []
        
        # X_last의 각 컬럼을 순회하며 해당 컬럼에 맞는 스케일러 적용
        for col in X_last.columns:
            scaler = feature_scalers_dict.get(col)
            if scaler is None:
                raise ValueError(f"Scaler not found for: {col} (Target: {target})")
                
            # 한 컬럼씩 스케일링 후 리스트에 추가 (transform은 DataFrame 형태를 유지하여 적용)
            scaled_col = scaler.transform(X_last[[col]])
            scaled_features.append(scaled_col)
        
        # 스케일링된 컬럼들을 수평으로 합쳐서 하나의 2D 배열을 생성
        X_input_scaled_2d = np.hstack(scaled_features)
        
        # 모델 입력 형태 (1, LOOK_BACK, num_features)로 변경
        X_input_scaled = X_input_scaled_2d.reshape(1, LOOK_BACK, -1)
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
