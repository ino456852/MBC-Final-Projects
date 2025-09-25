import os
import tensorflow as tf
import json
from .models import MODELS_TO_TRAIN
from .constant import (
    get_model_config,
    BEST_PARAMS_PTH,
    MODELS_DIR,
    TARGETS,
)
from .data import get_master_dataframe, prepare_data_for_target


def run_training():
    if not os.path.exists(BEST_PARAMS_PTH):
        print(f"[{BEST_PARAMS_PTH}] 파일을 찾을 수 없습니다.")

    with open(BEST_PARAMS_PTH, "r") as f:
        all_best_params = json.load(f)

    print("데이터 로딩 및 전처리 시작...")
    data_cpy = get_master_dataframe()

    os.makedirs(MODELS_DIR, exist_ok=True)

    for target_name in TARGETS:
        print(f"\n===== '{target_name}' 모델 훈련 시작 =====")

        best_params = all_best_params.get(target_name, {})
        best_model_name = best_params.get(
            "model_name", "mha"
        )  # 성능이 가장 좋은 매커니즘이여서 기준 파라미터로 사용
        print(f"최적 모델({best_model_name}) 기준 파라미터로 모든 모델을 훈련합니다.")

        train_config = get_model_config()
        train_config.update(best_params)

        X_train, y_train, _, _, _, _ = prepare_data_for_target(data_cpy, target_name)

        num_features = X_train.shape[2]

        print("X_train_mean:", X_train.mean())

        for model_key, build_fn in MODELS_TO_TRAIN.items():
            print(f"  -> {model_key} 모델 처리 중...")

            save_path = os.path.join(
                MODELS_DIR, f"{target_name}_{model_key}_attention.keras"
            )
            if os.path.exists(save_path):
                print(f"{model_key} -> 이미 훈련된 모델 발견. 건너뜁니다.")
                continue

            print(f"{model_key} -> 모델 훈련 시작...")
            model = build_fn(train_config.LOOK_BACK, num_features, params=train_config)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=train_config.PATIENCE,
                    restore_best_weights=True,
                )
            ]
            model.fit(
                X_train,
                y_train,
                epochs=train_config.EPOCHS,
                batch_size=train_config.BATCH_SIZE,
                validation_split=train_config.VALIDATION_SPLIT,
                callbacks=callbacks,
                shuffle=False,
                verbose=0,
            )

            model.save(save_path)
            print(f"  -> 모델 훈련 및 저장 완료: {save_path}")

    print("\n===== 모든 모델 훈련/확인 작업이 완료되었습니다. =====")


if __name__ == "__main__":
    run_training()
