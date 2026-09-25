from pathlib import Path

from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from tqdm import tqdm

SEED = 20260916
DEVICE = "cpu"  # "cpu" или "cuda"
DATA_DIR = Path(__file__).resolve().parent
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

# Выделение признаков (исключаем row_id и target)
feature_columns = [col for col in train.columns if col not in {"row_id", "target"}]

X_train = train[feature_columns]
y_train = train["target"]
X_validation = validation[feature_columns]
y_validation = validation["target"]
X_test = test[feature_columns]

print(
    f"Обучающая выборка: {X_train.shape}, "
    f"Валидация: {X_validation.shape}, "
    f"Тест: {X_test.shape}"
)


param_grid = {
    "boosting_type": ["gbdt", "goss"],
    "max_depth": [4, 5, 6, 7],
    "num_leaves": [31, 63, 95, 127],
    "min_child_samples": [20, 25, 28, 30, 40],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 300, 400, 500],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 0.05, 0.1, 0.5],
}

cv = list(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train)
)


def test_model(params):
    device = "cpu" if params.get("boosting_type") == "goss" else DEVICE
    n_jobs = -1 if device == "cpu" else 1
    model = LGBMClassifier(
        **params,
        device=device,
        random_state=SEED,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    )
    return scores.mean(), params


def run_grid(grid):
    print(f"Всего комбинаций для перебора: {len(grid)}")
    tasks = [delayed(test_model)(p) for p in grid]
    return list(
        tqdm(
            Parallel(n_jobs=4, batch_size="auto", return_as="generator")(tasks),
            total=len(tasks),
            desc="Прогресс",
        )
    )


coarse_grid = param_grid.copy()
coarse_grid.update(
    max_depth=[5],
    num_leaves=[63, 95],
    min_child_samples=[20, 30],
    n_estimators=[300, 400],
    learning_rate=[0.05, 0.1],
    colsample_bytree=[0.8],
    reg_lambda=[0.0, 0.1],
)
print("Грубый перебор гиперпараметров")
results = run_grid(list(ParameterGrid(coarse_grid)))

top_results = sorted(results, key=lambda x: x[0], reverse=True)[:3]
coarse_best_auc, coarse_best_params = top_results[0]
print(f"Лучший AUC после 1-го этапа: {coarse_best_auc:.5f}")
print(f"Лучшая комбинация: {coarse_best_params}")

# Здесь только радиусы числовых окон
window = {
    "max_depth": 1,
    "num_leaves": 32,
    "min_child_samples": 5,
    "learning_rate": 0.03,
    "n_estimators": 100,
    "colsample_bytree": 0.15,
    "reg_lambda": 0.1,
}
fine_grid = []
checked = {tuple(sorted(params.items())) for _, params in results}
for _, params in top_results:
    # Сначала сохраняем ВСЕ параметры; затем уточняем только числовые значения.
    local_grid = {name: [value] for name, value in params.items()}
    for name, radius in window.items():
        if isinstance(params[name], (int, float)):
            local_grid[name] = [
                value
                for value in param_grid[name]
                if isinstance(value, (int, float))
                and round(abs(value - params[name]), 10) <= radius
            ]
    for p in ParameterGrid(local_grid):
        key = tuple(sorted(p.items()))
        if key not in checked:
            checked.add(key)
            fine_grid.append(p)

print("Уточнение гиперпараметров")
results.extend(run_grid(fine_grid))

best_auc, best_params = max(results, key=lambda x: x[0])
print(f"Лучший средний AUC по кросс-валидации: {best_auc:.5f}")
print(f"Лучшая комбинация: {best_params}")
for param_name, param_val in best_params.items():
    print(f"  {param_name}: {param_val}")

device = "cpu" if best_params.get("boosting_type") == "goss" else DEVICE
model = LGBMClassifier(
    **best_params,
    device=device,
    random_state=SEED,
    n_jobs=4 if device == "cpu" else 1,
    verbosity=-1,
)
model.fit(X_train, y_train)
probs = model.predict_proba(X_validation)[:, 1]
print(f"AUC на валидации: {roc_auc_score(y_validation, probs):.5f}")
