import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

PATH = "D:\СШІ\Lab2\Task2\income_data.txt"
SEED = 42
LIMIT = None

def to_floatable(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False

t0 = time.time()
X_raw, y_raw = [], []

with open(PATH, "r", encoding="utf-8") as f:
    for i, row in enumerate(f):
        if LIMIT and i >= LIMIT:
            break
        parts = row.strip().split(",")
        if len(parts) != 15 or "?" in parts:
            continue
        X_raw.append(parts[:-1])
        y_raw.append(parts[-1])

print(f"Завантажено {len(X_raw)} рядків за {time.time()-t0:.2f} сек.")

X_raw = np.array(X_raw, dtype=object)
y_raw = np.array(y_raw)

X_enc = np.zeros(X_raw.shape, dtype=float)
enc_list = []

for j in range(X_raw.shape[1]):
    col = X_raw[:, j]
    if all(to_floatable(v) for v in col):
        X_enc[:, j] = col.astype(float)
        enc_list.append(None)
    else:
        le = LabelEncoder()
        X_enc[:, j] = le.fit_transform(col)
        enc_list.append(le)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SVC(
    kernel="sigmoid",
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