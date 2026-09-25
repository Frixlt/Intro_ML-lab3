from pathlib import Path

from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from tqdm import tqdm

SEED = 20260916
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
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [7, 15, 31, 47, 63],
    "min_child_samples": [10, 20, 30, 40, 50],
}

cv = list(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train)
)


def test_model(params):
    model = LGBMClassifier(
        **params,
        random_state=SEED,
        n_jobs=1,
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
            Parallel(n_jobs=-1, batch_size=64, return_as="generator")(tasks),
            total=len(tasks),
            desc="Прогресс",
        )
    )


coarse_grid = param_grid.copy()
coarse_grid.update(
    n_estimators=[100, 300, 500],
    learning_rate=[0.03, 0.1],
    num_leaves=[7, 31, 63],
    min_child_samples=[10, 30, 50],
)
print("Грубый перебор гиперпараметров")
results = run_grid(list(ParameterGrid(coarse_grid)))

top_results = sorted(results, key=lambda x: x[0], reverse=True)[:3]
# Здесь только радиусы числовых окон
window = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 16,
    "min_child_samples": 10,
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
    for params in ParameterGrid(local_grid):
        key = tuple(sorted(params.items()))
        if key not in checked:
            checked.add(key)
            fine_grid.append(params)

print("Уточнение гиперпараметров")
results.extend(run_grid(fine_grid))

best_auc, best_params = max(results, key=lambda x: x[0])
print(f"Лучший средний AUC по кросс-валидации: {best_auc:.5f}")
for param_name, param_val in best_params.items():
    print(f"  {param_name}: {param_val}")

# финальное обучение
model = LGBMClassifier(**best_params, random_state=SEED, n_jobs=1, verbosity=-1)
model.fit(X_train, y_train)
probs = model.predict_proba(X_validation)[:, 1]
print(f"AUC на валидации: {roc_auc_score(y_validation, probs):.5f}")
