import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

DATASET = "D:\СШІ\Lab2\Task2\income_data.txt"
SEED = 42
LIMIT_ROWS = None

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

t0 = time.time()
X_list, y_list = [], []

with open(DATASET, "r", encoding="utf-8") as fh:
    for i, row in enumerate(fh):
        if LIMIT_ROWS and i >= LIMIT_ROWS:
            break
        parts = row.strip().split(",")
        if len(parts) != 15 or "?" in parts:
            continue
        X_list.append(parts[:-1])
        y_list.append(parts[-1])

print(f"Завантажено {len(X_list)} рядків (час {time.time()-t0:.2f} сек.)")

X_list = np.array(X_list, dtype=object)
y_list = np.array(y_list)

X_proc = np.zeros(X_list.shape, dtype=float)
enc_list = []

for k in range(X_list.shape[1]):
    col = X_list[:, k]
    if all(is_number(v) for v in col):
        X_proc[:, k] = col.astype(float)
        enc_list.append(None)
    else:
        le = LabelEncoder()
        X_proc[:, k] = le.fit_transform(col)
        enc_list.append(le)

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y_list, test_size=0.2, random_state=SEED, stratify=y_list
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=SEED
)

t1 = time.time()
clf.fit(X_train, y_train)
print(f"Навчання завершено за {time.time()-t1:.2f} сек.")

y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall:", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1:", round(f1_score(y_test, y_pred, average="weighted"), 4))

print("\n=== Звіт класифікації ===")
print(classification_report(y_test, y_pred))

print("=== Матриця плутанини ===")
print(confusion_matrix(y_test, y_pred))