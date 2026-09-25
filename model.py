from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# criterion = ["gini", "log_loss"]
# max_features = [None, "sqrt", 0.5]

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
print(depth, leaves)
best_auc = 0
best_params = None

for d in depth:
    for leave in leaves:
        my_tree = DecisionTreeClassifier(
            max_depth=d, min_samples_leaf=leave, random_state=SEED
        )
        my_tree.fit(X_train, y_train)
        tree_probs = my_tree.predict_proba(X_validation)[:, 1]

        auc = roc_auc_score(y_validation, tree_probs)
        if auc > best_auc:
            best_auc = auc
            best_params = (d, leave)

print(f"Лучший ROC-AUC: {best_auc}, параметры (depth, leaf): {best_params}")
