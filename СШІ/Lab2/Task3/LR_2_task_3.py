import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris_data = load_iris()
features, targets = iris_data.data, iris_data.target

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42, stratify=targets
)

clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("\n=== Звіт класифікації ===")
print(classification_report(y_test, predictions, target_names=iris_data.target_names))

print("=== Матриця плутанини ===")
print(confusion_matrix(y_test, predictions))