from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
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
    "max_depth": list(range(1, 13)) + [None],
    "min_samples_split": list(range(2, 21, 2)),
    "min_samples_leaf": list(range(1, 36, 2)),
    "max_features": [None, "sqrt", "log2"]
    + [round(x * 0.1, 1) for x in range(1, 10, 2)],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0] + [round(x * 0.002, 4) for x in range(1, 6)],
}

grid = list(ParameterGrid(param_grid))

print(f"Всего комбинаций для перебора: {len(grid)}")


def test_tree(params):
    tree = DecisionTreeClassifier(**params, random_state=SEED)
    tree.fit(X_train, y_train)
    probs = tree.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, probs)
    return auc, params


tasks = [delayed(test_tree)(p) for p in grid]

print("Перебор гиперпараметров")

results = list(
    tqdm(
        Parallel(n_jobs=-1, batch_size=64, return_as="generator")(tasks),
        total=len(tasks),
        desc="Прогресс",
    )
)

best_auc, best_params = max(results, key=lambda x: x[0])
print(f"Лучший AUC: {best_auc:.5f}")
for param_name, param_val in best_params.items():
    print(f"  {param_name}: {param_val}")
