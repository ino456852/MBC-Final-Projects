import os
import numpy as np
from itertools import product
from .constant import TARGETS, LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE
from .preprocess import load_preprocess_data
from ..attention_lstm_rolling_new.model import build_model
import tensorflow as tf


# rolling window 방식으로 학습/테스트 인덱스를 생성하는 함수
def rolling_split_indices(total_len, train_size=1200, test_size=300):
    """
    전체 데이터 길이에서 rolling window 방식으로
    train/test 인덱스 배열을 생성합니다.
    """
    for start in range(0, total_len - train_size - test_size + 1, test_size):
        yield (
            np.arange(start, start + train_size),  # 학습 데이터 인덱스
            np.arange(
                start + train_size, start + train_size + test_size
            ),  # 테스트 데이터 인덱스
        )


# 모델을 학습하고 저장하는 메인 함수
def train_and_save_models(alphas=None):
    """
    각 타겟별로 rolling window cross-validation을 통해
    최적의 하이퍼파라미터를 찾고, 전체 데이터로 최종 모델을 학습하여 저장합니다.
    """
    if alphas is None:
        alphas = {5: 0.1, 20: 0.1, 60: 0.1, 120: 0.1}  # 기본 알파값 설정

    # 하이퍼파라미터 후보군 정의
    HYPERPARAMS_CANDIDATES = {
        "lstm_units": [100, 150],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
    }
    all_models_info = {}  # 각 타겟별 모델 정보 저장용

    for target in TARGETS:
        print(f"= {target.upper()} 모델 학습 시작 =")
        data = load_preprocess_data(target, alphas)  # 데이터 전처리 및 로드
        df = data["df"]
        features = data["features"]
        X_seq = data["X_seq"]  # 시계열 입력 데이터
        y_seq = data["y_seq"]  # 타겟 시계열 데이터
        val_losses_per_param = {}  # 하이퍼파라미터별 검증 손실 저장
        total_len = len(df)

        # rolling window cross-validation
        for train_idx, test_idx in rolling_split_indices(total_len):
            X_train, y_train = X_seq[train_idx], y_seq[train_idx]
            X_test, y_test = X_seq[test_idx], y_seq[test_idx]
            best_fold_val_loss = float("inf")  # fold별 최적 검증 손실
            best_fold_params = None  # fold별 최적 하이퍼파라미터

            # 모든 하이퍼파라미터 조합에 대해 학습 및 검증
            for lstm_units, dropout_rate, learning_rate in product(
                HYPERPARAMS_CANDIDATES["lstm_units"],
                HYPERPARAMS_CANDIDATES["dropout_rate"],
                HYPERPARAMS_CANDIDATES["learning_rate"],
            ):
                # 모델 생성
                model = build_model(
                    LOOK_BACK, X_train.shape[2], lstm_units, dropout_rate, learning_rate
                )
                # 조기 종료 콜백 설정
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
                # 모델 학습
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=30,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0,
                )
                val_loss = min(history.history["val_loss"])  # 최소 검증 손실
                # fold 내에서 가장 좋은 하이퍼파라미터 저장
                if val_loss < best_fold_val_loss:
                    best_fold_val_loss = val_loss
                    best_fold_params = (lstm_units, dropout_rate, learning_rate)
            # fold별 최적 하이퍼파라미터의 검증 손실 저장
            val_losses_per_param.setdefault(best_fold_params, []).append(
                best_fold_val_loss
            )

        # 각 하이퍼파라미터 조합별 평균 검증 손실 계산
        avg_val_losses = {
            params: np.mean(losses) for params, losses in val_losses_per_param.items()
        }
        # 평균 검증 손실이 가장 낮은 하이퍼파라미터 선택
        best_params = min(avg_val_losses, key=avg_val_losses.get)
        print(
            f"{target.upper()} 최적 하이퍼파라미터: {best_params}, 평균 val_loss={avg_val_losses[best_params]:.4f}"
        )

        # 전체 데이터로 최종 모델 재학습
        final_model = build_model(LOOK_BACK, X_seq.shape[2], *best_params)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        final_model.fit(
            X_seq, y_seq, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0
        )
        # 모델 저장
        model_name = f"{target}{KERAS_FILE_TEMPLATE}"
        final_model.save(os.path.join(MODEL_DIR, f"{model_name}"))
        # 모델 및 스케일러 등 정보 저장
        all_models_info[target] = {
            "model": final_model,
            "feature_scaler": data["feature_scaler"],
            "target_scaler": data["target_scaler"],
            "features": features,
        }
    return all_models_info


# 메인 실행부
if __name__ == "__main__":
    print("학습 시작")
    train_and_save_models()
    print("학습 완료")
