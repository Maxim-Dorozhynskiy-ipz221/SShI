import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

path = "D:\СШІ\Lab2\Task1\income_data.txt"
X_raw, y_raw = [], []
cap = 25000
low, high = 0, 0

with open(path, "r") as f:
    for line in f:
        if low >= cap and high >= cap:
            break
        if "?" in line:
            continue
        parts = line.strip().split(", ")
        target = parts[-1]
        if target == "<=50K" and low < cap:
            X_raw.append(parts[:-1])
            y_raw.append(target)
            low += 1
        elif target == ">50K" and high < cap:
            X_raw.append(parts[:-1])
            y_raw.append(target)
            high += 1

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

X_encoded = np.zeros_like(X_raw, dtype=int)
transformers = []

for j in range(X_raw.shape[1]):
    col = X_raw[:, j]
    try:
        X_encoded[:, j] = col.astype(int)
        transformers.append(None)
    except ValueError:
        le = LabelEncoder()
        X_encoded[:, j] = le.fit_transform(col)
        transformers.append(le)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_raw, test_size=0.2, random_state=42
)

clf = OneVsOneClassifier(LinearSVC(random_state=0))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, pos_label=">50K") * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred, pos_label=">50K") * 100, 2), "%")
print("F1-score:", round(f1_score(y_test, y_pred, pos_label=">50K") * 100, 2), "%")

sample = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
    'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
    '0', '0', '40', 'United-States'
]

sample_encoded = []
for idx, val in enumerate(sample):
    enc = transformers[idx]
    if enc is None:
        sample_encoded.append(int(val))
    else:
        if val in enc.classes_:
            sample_encoded.append(enc.transform([val])[0])
        else:
            print(f"Попередження: '{val}' відсутнє у тренувальних даних для стовпця {idx}.")
            sample_encoded.append(-1)

sample_encoded = np.array(sample_encoded).reshape(1, -1)
pred = clf.predict(sample_encoded)
print("Prediction for new sample:", pred[0])