import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

def read_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не знайдено.")
    arr = np.loadtxt(path, delimiter=',', dtype=float)
    X = arr[:, :2]
    y = arr[:, 2].astype(int)
    return X, y

def draw_surface(ax, model, X, y, title):
    pad = 0.7
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25)

    markers = {0: 'o', 1: 's'}
    for lab, mk in markers.items():
        ax.scatter(X[y == lab, 0], X[y == lab, 1],
                   s=45, marker=mk, label=f'class {lab}', edgecolor='k')

    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

def build_model(X_train, y_train, balanced: bool):
    params = dict(n_estimators=100, max_depth=7, random_state=1)
    if balanced:
        clf = ExtraTreesClassifier(class_weight="balanced", **params)
        suffix = " (balanced)"
    else:
        clf = ExtraTreesClassifier(**params)
        suffix = " (unbalanced)"
    clf.fit(X_train, y_train)
    return clf, suffix

def assess_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

def main():
    parser = argparse.ArgumentParser(description="Task 2.2 — Class imbalance")
    parser.add_argument("--data", type=str, default="D:\СШІ\Lab5\data_imbalance.txt")
    parser.add_argument("--balance", type=str, choices=["off", "on"], default="off")
    parser.add_argument("--ignore", action="store_true")
    args = parser.parse_args()

    if args.ignore:
        warnings.filterwarnings("ignore")

    X, y = read_data(Path(args.data))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    model, suffix = build_model(X_train, y_train, balanced=(args.balance == "on"))

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_surface(ax, model, X_train, y_train, f"Decision boundary{suffix}")
    plt.tight_layout()
    plt.show()

    assess_model(model, X_test, y_test)

if __name__ == "__main__":
    main()