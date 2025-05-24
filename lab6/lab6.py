import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y = np.array([1.15, 1.39, 1.85, 1.95, 2.16, 2.79, 2.88, 2.38, 3.31])

n = len(x)

Sx = np.sum(x)
Sy = np.sum(y)
Sxx = np.sum(x ** 2)
Sxy = np.sum(x * y)

den = n * Sxx - Sx ** 2

a = (n * Sxy - Sx * Sy) / den
b = (Sxx * Sy - Sx * Sxy) / den

print(f"Коэффициенты линейной аппроксимации: a = {a:.4f}, b = {b:.4f}")

y_pred = a * x + b
eps = y - y_pred

delta = np.sqrt(np.sum(eps ** 2) / n)
print(f"Среднеквадратичное отклонение delta = {delta:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(x, y, marker='o', label='Табличные точки')
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, a * x_line + b, label=f'Аппроксимация: y={a:.4f}x+{b:.4f}')

plt.title('Линейная аппроксимация методом МНК')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
