from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
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
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": list(range(2, 21)),
    "min_samples_split": list(range(2, 31)),
    "min_samples_leaf": list(range(1, 41)),
    "max_features": [None, "sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01],
}

cv = list(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train)
)


def test_tree(params):
    tree = DecisionTreeClassifier(**params, random_state=SEED)
    scores = cross_val_score(
        tree,
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
    tasks = [delayed(test_tree)(p) for p in grid]
    return list(
        tqdm(
            Parallel(n_jobs=-1, batch_size=64, return_as="generator")(tasks),
            total=len(tasks),
            desc="Прогресс",
        )
    )


coarse_grid = param_grid.copy()
coarse_grid.update(
    max_depth=[2, 8, 14, 20],
    min_samples_split=[2, 16, 30],
    min_samples_leaf=[1, 20, 40],
    ccp_alpha=[0.0, 0.005, 0.01],
)
print("Грубый перебор гиперпараметров")
results = run_grid(list(ParameterGrid(coarse_grid)))

top_results = sorted(results, key=lambda x: x[0], reverse=True)[:3]
# Здесь только радиусы числовых окон
window = {
    "max_depth": 3,
    "min_samples_split": 8,
    "min_samples_leaf": 10,
    "max_features": 0.2,
    "ccp_alpha": 0.004,
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
tree = DecisionTreeClassifier(**best_params, random_state=SEED)
tree.fit(X_train, y_train)
probs = tree.predict_proba(X_validation)[:, 1]
print(f"AUC на валидации: {roc_auc_score(y_validation, probs):.5f}")
