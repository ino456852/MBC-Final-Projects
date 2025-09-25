import optuna
import tensorflow as tf
import numpy as np
from functools import partial
import json
from .data_processor import DataProcessor
from .constant import LOOK_BACK, BEST_PARAMS_PTH
from .models import MODELS_TO_TRAIN


def objective(trial: int, dataprocessor: DataProcessor, target_currency: str):
    print(f"\n===== Trial #{trial.number} ({target_currency.upper()}) 시작 =====")

    model_name = trial.suggest_categorical("model_name", MODELS_TO_TRAIN.keys())
    build_fn = MODELS_TO_TRAIN[model_name]
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", ["AdamW", "Nadam", "Adadelta", "Adafactor"]
    )

    default_params = {
        "lstm_units": 150,
        "lstm_units_half": 75,
        "num_heads": 8,
        "key_dim": 64,
    }

    default_params[optimizer_name] = optimizer_name
    default_params["lstm_units"] = trial.suggest_int("lstm_units", 100, 200, step=50)
    default_params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    if optimizer_name in ["AdamW", "Nadam"]:
        default_params["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-4, 1e-3, log=True
        )
    else:
        default_params["learning_rate"] = 0.001

    if model_name in ["mha", "cross", "causal"]:
        default_params["num_heads"] = trial.suggest_categorical("num_heads", [8, 12])
        default_params["key_dim"] = trial.suggest_categorical("key_dim", [32, 64])
    if model_name == "cross":
        default_params["lstm_units"] = trial.suggest_int(
            "lstm_units_half", 75, 125, step=25
        )

    X_train, y_train, _, _, _ = dataprocessor.get_sequence_data(target_currency)
    num_features = X_train.shape[2]

    if model_name == "cross":
        num_base_features = dataprocessor.num_base_features
        model = build_fn(
            LOOK_BACK, num_base_features, num_features, params=default_params
        )
    else:
        model = build_fn(LOOK_BACK, num_features, params=default_params)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    val_loss = np.min(history.history["val_loss"])
    print(
        f"Trial #{trial.number} ({target_currency.upper()}) 완료. 모델: {model_name.upper()}, 옵티마이저: {optimizer_name}, Val Loss: {val_loss:.6f}"
    )

    tf.keras.backend.clear_session()
    return val_loss


def run_optuna():
    print("전체 통화 자동 튜닝 (옵티마이저 포함)을 시작합니다...")
    dataprocessor = DataProcessor()

    all_best_params = {}
    n_trials_per_currency = 20

    for currency in dataprocessor.targets:
        print(f"\n{'=' * 20} {currency.upper()} 통화 튜닝 시작 {'=' * 20}")
        study = optuna.create_study(
            study_name=f"{currency}_study", direction="minimize"
        )

        try:
            study.optimize(
                partial(
                    objective, dataprocessor=dataprocessor, target_currency=currency
                ),
                n_trials=n_trials_per_currency,
            )
        except KeyboardInterrupt:
            print(f"{currency.upper()} 튜닝이 사용자에 의해 중단되었습니다.")

        all_best_params[currency] = study.best_params
        print(f"\n--- {currency.upper()} 통화 최적 결과 ---")
        print(study.best_params)

    with open(BEST_PARAMS_PTH, "w", encoding="utf-8") as f:
        json.dump(all_best_params, f, ensure_ascii=False, indent=4)

    print(f"\n{'=' * 20} 전체 튜닝 완료 {'=' * 20}")
    print(f"결과가 '{BEST_PARAMS_PTH}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    # best_params.json 만드는 함수
    run_optuna()
