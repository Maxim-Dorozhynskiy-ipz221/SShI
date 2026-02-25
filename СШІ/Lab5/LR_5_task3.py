import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "D:\СШІ\Lab5\data_random_forests.txt"

def read_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не знайдено.")
    arr = np.loadtxt(path, delimiter=',', dtype=float)
    X = arr[:, :2]
    y = arr[:, 2].astype(int)
    return X, y

def grid_search(X, y, metric: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    model = RandomForestClassifier(random_state=1)

    params = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 7]
    }

    search = GridSearchCV(
        model,
        param_grid=params,
        scoring=metric,
        cv=5,
        n_jobs=-1,
        return_train_score=True
    )
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)

    print(f"\n=== Metric: {metric} ===")
    print("Best parameters:", search.best_params_)
    print("Best CV score:", search.best_score_)
    print("\nReport on test data:\n", classification_report(y_test, y_pred, digits=4))

    results = pd.DataFrame(search.cv_results_)
    results = results[["params", "mean_test_score", "std_test_score", "mean_train_score"]]
    print("\nGrid summary (top 5 rows):")
    print(results.sort_values("mean_test_score", ascending=False).head())

def main():
    X, y = read_dataset(Path(DATA_PATH))
    for metric in ["precision_macro", "recall_macro"]:
        grid_search(X, y, metric)

if __name__ == "__main__":
    main()