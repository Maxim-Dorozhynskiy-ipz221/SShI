import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

DATA_PATH = "D:\СШІ\Lab2\Task4\income_data.txt"
SEED = 42

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False

def load_dataset(path: str):
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            parts = row.strip().split(",")
            if len(parts) != 15 or "?" in parts:
                continue
            X.append(parts[:-1])
            y.append(parts[-1])
    X = np.array(X, dtype=object)
    y = np.array(y)

    X_enc = np.zeros(X.shape, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        if all(is_number(v) for v in col):
            X_enc[:, j] = col.astype(float)
        else:
            le = LabelEncoder()
            X_enc[:, j] = le.fit_transform(col)
    return X_enc, y

def evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "rec": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

X, y = load_dataset(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "DecisionTree": DecisionTreeClassifier(random_state=SEED),
    "NaiveBayes": GaussianNB(),
    "SVM_RBF": SVC(kernel="rbf", gamma="scale", random_state=SEED),
}

results = {name: evaluate(clf, X_train, X_test, y_train, y_test) for name, clf in models.items()}

print("\n=== Порівняння алгоритмів ===")
for name, metrics in sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True):
    print(f"{name:18s} acc={metrics['acc']:.4f} prec={metrics['prec']:.4f} rec={metrics['rec']:.4f} f1={metrics['f1']:.4f}")