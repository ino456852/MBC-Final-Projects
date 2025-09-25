import optuna
import tensorflow as tf
import numpy as np
from functools import partial
import json
from data import get_master_dataframe, prepare_data_for_target
from constant import get_model_config, BEST_PARAMS_PTH, TARGETS
from models import MODELS_TO_TRAIN


def objective(trial, data, target_currency):
    print(f"\n===== Trial #{trial.number} ({target_currency.upper()}) 시작 =====")

    model_name = trial.suggest_categorical("model_name", MODELS_TO_TRAIN.keys())
    build_fn = MODELS_TO_TRAIN[model_name]
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", ["AdamW", "Nadam", "Adadelta", "Adafactor"]
    )

    trial_config = get_model_config()
    trial_config[optimizer_name] = optimizer_name
    trial_config["LSTM_UNITS"] = trial.suggest_int("LSTM_UNITS", 50, 250, step=25)
    trial_config["DROPOUT_RATE"] = trial.suggest_float("DROPOUT_RATE", 0.1, 0.5)

    if optimizer_name in ["AdamW", "Nadam"]:
        trial_config["LEARNING_RATE"] = trial.suggest_float(
            "LEARNING_RATE", 1e-4, 1e-2, log=True
        )

    if model_name in ["mha", "cross", "causal"]:
        trial_config["NUM_HEADS"] = trial.suggest_categorical("NUM_HEADS", [4, 8, 16])
        trial_config["KEY_DIM"] = trial.suggest_categorical("KEY_DIM", [32, 64])
    if model_name == "cross":
        trial_config["LSTM_UNITS_HALF"] = trial.suggest_int(
            "LSTM_UNITS_HALF", 25, 125, step=25
        )

    X_train, y_train, _, _, _, _ = prepare_data_for_target(data, target_currency)
    num_features = X_train.shape[2]
    model = build_fn(trial_config["LOOK_BACK"], num_features, params=trial_config)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=trial_config["PATIENCE"], restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=trial_config["EPOCH"],
        batch_size=trial_config["BATCH_SIZE"],
        validation_split=trial_config["VALIDATION_SPLIT"],
        shuffle=False,
        callbacks=[early_stopping],
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
    master_data = get_master_dataframe()

    all_best_params = {}
    n_trials_per_currency = 25

    for currency in TARGETS:
        print(f"\n{'=' * 20} {currency.upper()} 통화 튜닝 시작 {'=' * 20}")
        study = optuna.create_study(
            study_name=f"{currency}_study", direction="minimize"
        )

        try:
            study.optimize(
                partial(objective, data=master_data, target_currency=currency),
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
