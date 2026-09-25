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

my_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED)
my_tree.fit(X_train, y_train)
tree_probs = my_tree.predict_proba(X_validation)[:, 1]
print("ROC-AUC дерева:", roc_auc_score(y_validation, tree_probs))
