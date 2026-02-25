import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

SEED = 0

iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

model = make_pipeline(
    StandardScaler(),
    RidgeClassifier(solver="sag", tol=1e-2, random_state=SEED)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1:", f1_score(y_test, y_pred, average="weighted"))
print("Cohen Kappa:", cohen_kappa_score(y_test, y_pred))
print("Matthews Corrcoef:", matthews_corrcoef(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred, target_names=names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names, yticklabels=names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("RidgeClassifier — Confusion Matrix (Iris)")
plt.tight_layout()
plt.show()