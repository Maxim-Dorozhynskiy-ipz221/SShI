import numpy as np
import matplotlib.pyplot as plt

X = np.array([9, 7, 4, 8], dtype=float)
Y = np.array([5, 1, 0, 6], dtype=float)

x_mean, y_mean = X.mean(), Y.mean()
slope = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean) ** 2)
intercept = y_mean - slope * x_mean

A = np.vstack([np.ones_like(X), X]).T
beta, *_ = np.linalg.lstsq(A, Y, rcond=None)

print(f"Аналітичний розрахунок: b0={intercept:.6f}, b1={slope:.6f}")
print(f"NumPy lstsq: b0={beta[0]:.6f}, b1={beta[1]:.6f}")

X_line = np.linspace(X.min(), X.max(), 300)
Y_line = intercept + slope * X_line

plt.scatter(X, Y, color="navy", label="Точки")
plt.plot(X_line, Y_line, color="crimson", label="Лінія МНК")
plt.title("Завдання 1 — Метод найменших квадратів")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.show()