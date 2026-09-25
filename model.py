from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import roc_auc_score
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


depth = list(range(1, 11))
leaves = list(range(5, 51, 5))
criterion = ["gini", "log_loss"]
max_features = [None, "sqrt", 0.5]
total_tasks = len(depth) * len(leaves) * len(criterion) * len(max_features)


def test_tree(**params):
    tree = DecisionTreeClassifier(**params, random_state=SEED)
    tree.fit(X_train, y_train)
    probs = tree.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, probs)
    return auc, params


tasks = [
    delayed(test_tree)(
        max_depth=d,
        min_samples_leaf=l,
        criterion=c,
        max_features=f,
    )
    for d in depth
    for l in leaves
    for c in criterion
    for f in max_features
]

results = list(
    tqdm(
        Parallel(n_jobs=-1, return_as="generator")(tasks),
        total=len(tasks),
        desc="Прогресс",
    )
)

best_auc, best_params = max(results, key=lambda x: x[0])

print(f"Лучший AUC: {best_auc:.5f}")
for param_name, param_val in best_params.items():
    print(f"  {param_name}: {param_val}")
