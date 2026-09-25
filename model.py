from pathlib import Path

from lightgbm import LGBMClassifier, early_stopping
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

# 1. Загрузка данных и фиксация seed
SEED = 20260916
DATA_DIR = Path(__file__).resolve().parent
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

# Выделение признаков (исключаем row_id и target)
feature_columns = [col for col in train.columns if col not in {"row_id", "target"}]

X_train = train[feature_columns].astype("float32")
y_train = train["target"].astype("float32")
X_validation = validation[feature_columns].astype("float32")
y_validation = validation["target"].astype("float32")
X_test = test[feature_columns].astype("float32")

print(
    f"Обучающая выборка: {X_train.shape}, "
    f"Валидация: {X_validation.shape}, "
    f"Тест: {X_test.shape}"
)


def stage1(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, step=0.01),
        "n_estimators": 1000,
        "random_state": SEED,
        "n_jobs": 2,
        "verbose": -1,
    }
    m = LGBMClassifier(**params).fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=40, verbose=False)],
    )
    trial.set_user_attr("best_iteration", m.best_iteration_)
    return roc_auc_score(y_validation, m.predict_proba(X_validation)[:, 1])


def stage2(trial, best_stage1):
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "goss"])
    params = {
        **best_stage1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.1),
        "boosting_type": boosting_type,
        "random_state": SEED,
        "n_estimators": 1000,
        "n_jobs": 2,
        "verbose": -1,
    }
    if boosting_type == "goss":
        params["subsample"] = 1.0
        params["subsample_freq"] = 0
    elif "subsample" not in params:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 0.9, step=0.1)
        params["subsample_freq"] = trial.suggest_int("subsample_freq", 1, 5)
    m = LGBMClassifier(**params).fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=40, verbose=False)],
    )
    trial.set_user_attr("best_iteration", m.best_iteration_)
    return roc_auc_score(y_validation, m.predict_proba(X_validation)[:, 1])


def stage3(trial, best_stage2):
    params = {
        **best_stage2,
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0, step=0.1),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1, step=0.02),
        "random_state": SEED,
        "n_estimators": 1000,
        "n_jobs": 2,
        "verbose": -1,
    }
    m = LGBMClassifier(**params).fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=40, verbose=False)],
    )
    trial.set_user_attr("best_iteration", m.best_iteration_)
    return roc_auc_score(y_validation, m.predict_proba(X_validation)[:, 1])


def cma_objective(trial):
    # t0 = time.perf_counter()
    params = {
        "boosting_type": "goss",
        "subsample": 1.0,
        "subsample_freq": 0,
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-3, 10.0, log=True
        ),
        "n_estimators": 1000,
        "random_state": SEED,
        "n_jobs": 2,
        "verbose": -1,
    }
    m = LGBMClassifier(**params).fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    trial.set_user_attr("best_iteration", m.best_iteration_)
    return roc_auc_score(y_validation, m.predict_proba(X_validation)[:, 1])


if __name__ == "__main__":
    # study_stage2 = optuna.create_study(
    #     direction="maximize",
    #     sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
    # )
    # study_stage2.optimize(
    #     lambda t: stage2(t, study_stage1.best_params), n_trials=1000, n_jobs=12
    # )
    # print(f"stage 1:{study_stage2.best_value}")
    # study_stage3 = optuna.create_study(
    #     direction="maximize",
    #     sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
    # )
    # study_stage3.optimize(
    #     lambda t: stage3(t, {**study_stage1.best_params, **study_stage2.best_params}),
    #     n_trials=1000,
    #     n_jobs=12,
    # )
    # print(f"stage 1:{study_stage3.best_value}")
    # for param_name, param_value in {
    #     **study_stage1.best_params,
    #     **study_stage2.best_params,
    #     **study_stage3.best_params,
    # }.items():
    #     print(f"{param_name:20s}: {param_value}")
    # total_start = time.perf_counter()
    # study = optuna.create_study(
    #     direction="maximize", sampler=optuna.samplers.CmaEsSampler(seed=SEED)
    # )
    # study.enqueue_trial(
    #     {
    #         "boosting_type": "goss",
    #         "max_depth": 5,
    #         "num_leaves": 95,
    #         "min_child_samples": 28,
    #         "learning_rate": 0.10,
    #         "colsample_bytree": 0.8,
    #         "reg_lambda": 0.1,
    #         "reg_alpha": 0.0,
    #         "min_split_gain": 0.0,
    #         "subsample": 1.0,
    #         "subsample_freq": 0,
    #     }
    # )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(
            seed=SEED,
            x0={
                "max_depth": 5,
                "num_leaves": 95,
                "min_child_samples": 28,
                "learning_rate": 0.10,
                "colsample_bytree": 0.80,
                "reg_lambda": 0.10,
                "reg_alpha": 0.00,
                "min_split_gain": 0.00,
                "min_child_weight": 0.4056,
            },
        ),
    )
    study.optimize(cma_objective, n_trials=300, n_jobs=1)
    # total_elapsed = time.perf_counter() - total_start
    print(f"ROC-AUC: {study.best_value:.5f}")
    for param_name, param_value in {
        **study.best_params,
    }.items():
        print(f"{param_name:20s}: {param_value}")

    # lr_model = LogisticRegression(
    #     C=0.01,
    #     max_iter=1000,
    #     random_state=SEED,
    # )
    # lr_model.fit(X_train, y_train)
    # val_preds_lr = lr_model.predict_proba(X_validation)[:, 1]
    # val_auc_lr = roc_auc_score(y_validation, val_preds_lr)
    # print(f"LogisticRegression ROC-AUC: {val_auc_lr:.5f}")
    # best_params_lgbmclassifier = {
    #     "max_depth": 5,
    #     "num_leaves": 95,
    #     "min_child_samples": 28,
    #     "learning_rate": 0.1,
    #     "n_estimators": 400,
    #     "boosting_type": "goss",
    #     "colsample_bytree": 0.8,
    #     "reg_lambda": 0.1,
    #     "reg_alpha": 0,
    #     "min_child_gain": 0,
    #     "min_child_weight": 0.4056303335131966,
    # }
    # lgbmc_model = LGBMClassifier(
    #     **best_params_lgbmclassifier, random_state=SEED, n_jobs=2, verbose=-1
    # )
    # lgbmc_model.fit(X_train, y_train)
    # val_preds_lgbmc = lgbmc_model.predict_proba(X_validation)[:, 1]
    # val_auc_lgbmc = roc_auc_score(y_validation, val_preds_lgbmc)
    # print(f"LGBMClassifier ROC-AUC: {val_auc_lgbmc:.5f}")

    # p_lgbmc_val = lgbmc_model.predict_proba(X_validation)[:, 1]
    # p_lr_val = lr_model.predict_proba(X_validation)[:, 1]
    # best = [0, 0]
    # for i in range(101):
    #     val_ensemble = (1 - (i / 100)) * p_lgbmc_val + (i / 100 * p_lr_val)
    #     auc_ensemble = roc_auc_score(y_validation, val_ensemble)
    #     if auc_ensemble > best[0]:
    #         best = [auc_ensemble, i]
    # print(best)
    # print(best[0] - val_auc_lgbmc)
    # test_preds = lgbmc_model.predict_proba(X_test)[:, 1]
    # submission = sample_submission.copy()
    # submission["target"] = test_preds
    # submission.to_csv(DATA_DIR / "submission.csv", index=False)
