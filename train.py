import os
import tensorflow as tf
import joblib
import json
from types import SimpleNamespace
import model_config as config
from training.data import get_master_dataframe, prepare_data_for_target, TARGETS
from training.models import MODELS_TO_TRAIN

def run_training():
    params_path = 'best_params.json'
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"'{params_path}' 파일을 찾을 수 없습니다.")
    
    with open(params_path, 'r') as f:
        all_best_params = json.load(f)

    print("데이터 로딩 및 전처리 시작...")
    master_data = get_master_dataframe()
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for target_name in TARGETS:
        print(f"\n===== '{target_name}' 모델 훈련 시작 =====")
        
        best_params = all_best_params.get(target_name, {})
        best_model_name = best_params.get('model_name', 'mha')
        print(f"최적 모델({best_model_name}) 기준 파라미터로 모든 모델을 훈련합니다.")
        
        train_config = SimpleNamespace(**vars(config))
        for key, value in best_params.items():
            setattr(train_config, key, value)

        X_train, y_train, _, _, feature_scaler, target_scaler = prepare_data_for_target(master_data, target_name)
        joblib.dump(feature_scaler, os.path.join(models_dir, f"{target_name}_feature_scaler.gz"))
        joblib.dump(target_scaler, os.path.join(models_dir, f"{target_name}_target_scaler.gz"))
        
        num_features = X_train.shape[2]
        
        for model_key, build_fn in MODELS_TO_TRAIN.items():
            model_name_str = model_key.replace('_', ' ').title()
            print(f"  -> {model_name_str} 모델 처리 중...")
            
            save_path = os.path.join(models_dir, f"{target_name}_{model_key}_attention.keras")
            if os.path.exists(save_path):
                print(f"  -> 이미 훈련된 모델 발견. 건너뜁니다.")
                continue
            
            print(f"  -> 모델 훈련 시작...")
            model = build_fn(train_config.LOOK_BACK, num_features, params=train_config)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=train_config.PATIENCE, restore_best_weights=True)
            model.fit(
                X_train, y_train, epochs=train_config.EPOCHS, batch_size=train_config.BATCH_SIZE,
                validation_split=train_config.VALIDATION_SPLIT, callbacks=[early_stopping], verbose=0
            )
            
            model.save(save_path)
            print(f"  -> 모델 훈련 및 저장 완료: {save_path}")

    print("\n===== 모든 모델 훈련/확인 작업이 완료되었습니다. =====")

if __name__ == '__main__':
    run_training()