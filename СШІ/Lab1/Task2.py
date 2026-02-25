import numpy as np
import matplotlib.pyplot as plt

def step(z: float) -> int:
    return 1 if z > 0 else 0

def xor_net(x1: int, x2: int):
    h1 = step(x1 - x2 - 0.5)
    h2 = step(x2 - x1 - 0.5)
    y = step(h1 + h2 - 0.5)
    return h1, h2, y

pts = np.array([[0,0],[0,1],[1,0],[1,1]])

gx = np.linspace(0,1,201)
gy = np.linspace(0,1,201)
X, Y = np.meshgrid(gx, gy)
Xb = (X >= 0.5).astype(int)
Yb = (Y >= 0.5).astype(int)
xor_map = (Xb ^ Yb).astype(int)

plt.figure(figsize=(7,7), dpi=130)
plt.title("Карта XOR", fontsize=15)
plt.pcolormesh(X, Y, xor_map, shading="nearest", cmap="coolwarm", alpha=0.8)
plt.scatter(pts[:,0], pts[:,1], c="black", s=80)
for px, py in pts:
    val = px ^ py
    plt.text(px+0.05, py+0.05, f"y={val}", fontsize=11, weight="bold")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

x_line = np.linspace(-0.1,1.1,200)
plt.figure(figsize=(9,9), dpi=150)
plt.title("Вхідний простір XOR", fontsize=17)
plt.plot(x_line, x_line-0.5, label="x₂ = x₁ - 0.5", linewidth=2)
plt.plot(x_line, x_line+0.5, label="x₂ = x₁ + 0.5", linewidth=2)

for px, py in pts:
    h1, h2, y = xor_net(px, py)
    plt.scatter(px, py, c="black", s=100)
    plt.text(px+0.04, py+0.04, f"h₁={h1}, h₂={h2}, y={y}",
             fontsize=13, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

hidden = np.array([xor_net(px, py) for px, py in pts])
layer = hidden[:, :2]
final = hidden[:, 2]

plt.figure(figsize=(7,7), dpi=130)
plt.title("Прихований шар XOR", fontsize=15)
colors = ["blue" if y==0 else "red" for y in final]
plt.scatter(layer[:,0], layer[:,1], c=colors, s=110, edgecolors="black")
for (h1,h2), y in zip(layer, final):
    plt.text(h1+0.03, h2+0.03, f"y={y}", fontsize=11,
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.plot(x_line, 0.5 - x_line, label="h₁ + h₂ = 0.5", linestyle="--", linewidth=2)
plt.xlabel("h₁")
plt.ylabel("h₂")
plt.legend()
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()