from pathlib import Path

from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 20260916
DATA_DIR = Path(__file__).resolve().parent

# Загрузка данных
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

feature_columns = [col for col in train.columns if col not in {"row_id", "target"}]

X_train = train[feature_columns]
y_train = train["target"]
X_validation = validation[feature_columns]
y_validation = validation["target"]
X_test = test[feature_columns]

# Лучшие параметры из model.py
best_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.7,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_samples": 20,
    "n_estimators": 500,
    "num_leaves": 31,
    "reg_lambda": 0.05,
}

print("Обучение модели...")
model = LGBMClassifier(
    **best_params,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)
model.fit(X_train, y_train)

# Оценка на отложенной валидации
val_probs = model.predict_proba(X_validation)[:, 1]
val_auc = roc_auc_score(y_validation, val_probs)
print(f"AUC на валидации: {val_auc:.5f}")

# Предсказание вероятностей для теста
test_probs = model.predict_proba(X_test)[:, 1]

submission = sample_submission.copy()
submission["target"] = test_probs
submission_path = DATA_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Файл {submission_path.name} успешно сохранён!")
print(submission.head())
