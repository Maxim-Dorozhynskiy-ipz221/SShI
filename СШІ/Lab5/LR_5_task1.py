import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

def read_csv(path: Path):
    data = np.loadtxt(path, delimiter=',', dtype=float)
    features = data[:, :2]
    labels = data[:, 2].astype(int)
    return features, labels

def show_points(ax, X, y, title):
    markers = ['o', 's', '^', 'x', 'D', 'P']
    for i, cls in enumerate(np.unique(y)):
        ax.scatter(
            X[y == cls, 0], X[y == cls, 1],
            s=45, marker=markers[i % len(markers)],
            label=f'class {cls}', edgecolor='k'
        )
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend(loc='best')

def show_surface(ax, model, X, y, title):
    pad = 0.7
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25)
    show_points(ax, X, y, title)

def make_model(kind: str, n_estimators: int, max_depth: int, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed
        ), "Random Forest"
    else:
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed
        ), "Extra Trees"

def assess(model, X_test, y_test, name: str):
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    test_points = np.array([
        [X_test[:, 0].mean(), X_test[:, 1].mean()],
        [X_test[:, 0].min(), X_test[:, 1].min()],
        [X_test[:, 0].max(), X_test[:, 1].max()]
    ])
    probs = model.predict_proba(test_points)
    print("\nProbability estimates for test points:")
    for i, p in enumerate(probs):
        print(f"  pt{i+1} {test_points[i]} -> {p}")

def main():
    parser = argparse.ArgumentParser(description="RandomForest vs ExtraTrees (Task 2.1)")
    parser.add_argument("--data", type=str, default="D:\СШІ\Lab5\data_random_forests.txt")
    parser.add_argument("--clf", type=str, choices=["rf", "erf"], default="rf")
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=1)
    args = parser.parse_args()

    X, y = read_csv(Path(args.data))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.random_state, stratify=y
    )

    model, name = make_model(args.clf, args.n_estimators, args.max_depth, args.random_state)
    model.fit(X_train, y_train)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    show_points(axs[0], X, y, "Input data (classes)")
    show_surface(axs[1], model, X_train, y_train, f"Decision boundaries — {name}")
    plt.tight_layout()
    plt.show()

    assess(model, X_test, y_test, name)

if __name__ == "__main__":
    main()