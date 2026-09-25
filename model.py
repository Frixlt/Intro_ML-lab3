from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

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


def test_tree(d, l):
    tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, random_state=SEED)
    tree.fit(X_train, y_train)
    probs = tree.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, probs)
    return auc, d, l


results = Parallel(n_jobs=-1)(delayed(test_tree)(d, l) for d in depth for l in leaves)
b_auc, b_depth, b_leaf = max(results, key=lambda x: x[0])
print(f"auc:{b_auc}, параметры: depth:{b_depth}, leaf:{b_leaf}")
