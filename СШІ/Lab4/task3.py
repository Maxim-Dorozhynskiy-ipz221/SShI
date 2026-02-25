import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.1, 0.3, 0.4, 0.6, 0.7], dtype=float)
Y = np.array([3.2, 3.0, 1.0, 1.8, 1.9], dtype=float)

poly = np.polyfit(X, Y, 4)

def f(val):
    return np.polyval(poly, val)

print("Коефіцієнти:", np.round(poly, 6))
print(f"f(0.2) = {f(0.2):.6f}")
print(f"f(0.5) = {f(0.5):.6f}")

X_line = np.linspace(0.05, 0.75, 400)
Y_line = f(X_line)

plt.scatter(X, Y, color="darkblue", label="Точки")
plt.plot(X_line, Y_line, color="orange", label="Поліном 4-го порядку")
plt.title("Завдання 3 — Інтерполяція")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()