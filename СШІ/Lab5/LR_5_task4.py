import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import __version__ as skl_version

def build_adaboost(base_model, n_estimators=200, lr=0.1, seed=1):
    try:
        return AdaBoostRegressor(
            estimator=base_model,
            n_estimators=n_estimators,
            learning_rate=lr,
            random_state=seed
        )
    except TypeError:
        return AdaBoostRegressor(
            base_estimator=base_model,
            n_estimators=n_estimators,
            learning_rate=lr,
            random_state=seed
        )

def check_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"R2={r2_score(y_test, y_pred):.4f}  "
          f"MAE={mean_absolute_error(y_test, y_pred):.4f}  "
          f"MSE={mean_squared_error(y_test, y_pred):.4f}")
    return y_pred

def show_importances(model, features):
    if not hasattr(model, "feature_importances_"):
        print("Модель не має атрибуту feature_importances_.")
        return
    imp = model.feature_importances_
    imp = imp / imp.sum()
    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(8, 4.5))
    plt.bar(range(len(imp)), imp[order])
    plt.xticks(range(len(imp)), [features[i] for i in order],
               rotation=45, ha='right')
    plt.ylabel("Відносна важливість")
    plt.title("Важливість ознак (AdaBoostRegressor)")
    plt.tight_layout()
    plt.show()

def main():
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )
    base_tree = DecisionTreeRegressor(max_depth=4, random_state=1)
    model = build_adaboost(base_tree, n_estimators=200, lr=0.1, seed=1)
    print(f"scikit-learn version: {skl_version}")
    model.fit(X_train, y_train)
    check_model(model, X_test, y_test)
    show_importances(model, feature_names)

if __name__ == "__main__":
    main()