import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

FILE = "D:\СШІ\Lab2\Task2\income_data.txt"
SEED = 42
LIMIT = None

def is_floatable(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False

t0 = time.time()
X_raw, y_raw = [], []

with open(FILE, "r") as f:
    for i, line in enumerate(f):
        if LIMIT and i >= LIMIT:
            break
        parts = line.strip().split(",")
        if len(parts) != 15 or "?" in parts:
            continue
        X_raw.append(parts[:-1])
        y_raw.append(parts[-1])

print(f"Залишилось {len(X_raw)} рядків після очищення (час {time.time()-t0:.2f}s)")

X_raw = np.array(X_raw, dtype=object)
y_raw = np.array(y_raw)

X_enc = np.zeros(X_raw.shape, dtype=float)
transformers = []

for j in range(X_raw.shape[1]):
    col = X_raw[:, j]
    if all(is_floatable(v) for v in col):
        X_enc[:, j] = col.astype(float)
        transformers.append(None)
    else:
        le = LabelEncoder()
        X_enc[:, j] = le.fit_transform(col)
        transformers.append(le)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVC(
        kernel="poly",
        degree=3,
        C=1.0,
        gamma="scale",
        random_state=SEED,
        cache_size=1000
    ))
])

t1 = time.time()
pipeline.fit(X_train, y_train)
print(f"Навчання завершено за {time.time()-t1:.2f}s")

y_pred = pipeline.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall:", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1:", round(f1_score(y_test, y_pred, average="weighted"), 4))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))