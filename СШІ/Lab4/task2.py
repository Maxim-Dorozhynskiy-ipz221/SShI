import numpy as np
import matplotlib.pyplot as plt

X = np.array([3.33, 1, 63, 0.87, 0.42, 0.27], dtype=float)
Y = np.array([0.48, 1.03, 2.02, 4.25, 7.16, 11.5], dtype=float)

coef, intercept = np.polyfit(X, Y, 1)
Y_pred = coef * X + intercept

def eval_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mse = np.mean((y_true - y_pred) ** 2)
    return r2, mse

r2_score, mse_score = eval_metrics(Y, Y_pred)

print(f"Рівняння: y = {intercept:.6f} + {coef:.6f}x")
print(f"R² = {r2_score:.6f}, MSE = {mse_score:.6f}")

X_line = np.linspace(X.min(), X.max(), 400)
Y_line = coef * X_line + intercept

plt.scatter(X, Y, color="blue", label="Дані")
plt.plot(X_line, Y_line, color="red", label="Регресія")
plt.title("Завдання 2 — Лінійна регресія")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()