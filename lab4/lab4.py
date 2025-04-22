import numpy as np
import matplotlib.pyplot as plt

A, B, C, D = 1, 3, -24, -10


def f(x):
    return A * x ** 3 + B * x ** 2 + C * x + D


def df(x):
    return 3 * A * x ** 2 + 2 * B * x + C


def ddf(x):
    return 6 * A * x + 2 * B


x_crit = np.array([-4, 2])
f_crit = f(x_crit)

x_inflection = -1
f_inflection = f(x_inflection)

print("Критические точки:")
for xi, yi in zip(x_crit, f_crit):
    print(f"x = {xi:.3f}, f(x) = {yi:.3f}")
print(f"\nТочка перегиба: x = {x_inflection:.3f}, f(x) = {f_inflection:.3f}\n")

x_vals = np.linspace(-10, 10, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='$f(x)=x^3+3x^2-24x-10$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.scatter(x_crit, f_crit, color='red', zorder=5, label='Критические точки')
plt.scatter(x_inflection, f_inflection, color='green', zorder=5, label='Точка перегиба')

plt.title('График функции f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

x_range = np.linspace(-10, 10, 400)
f_values = f(x_range)

intervals = []
for i in range(len(x_range) - 1):
    if f_values[i] * f_values[i + 1] < 0:
        intervals.append((x_range[i], x_range[i + 1]))

print("Функция меняет знак на:")
for interval in intervals:
    print(f"[{interval[0]:.3f}, {interval[1]:.3f}]")


def bisection_method(f, a, b, tol=1e-3, max_iter=1000):
    iterations = 0
    if f(a) * f(b) >= 0:
        raise ValueError("Функция не меняет знак на данном отрезке.")
    while abs(b - a) > tol and iterations < max_iter:
        c = (a + b) / 2.0
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iterations += 1
    return (a + b) / 2.0, iterations


bisection_results = []
print("\nМетод деления отрезка пополам:")
for a, b in intervals:
    try:
        root, iters = bisection_method(f, a, b, tol=1e-3)
        bisection_results.append((root, iters))
        print(
            f"Интервал [{a:.3f}, {b:.3f}]: найденный корень x = {root:.4f} за {iters} итераций (f(x) = {f(root):.4e})")
    except Exception as e:
        print(f"Ошибка в методе деления отрезков пополам: {e}")


def newton_method(f, df, x0, tol=1e-3, max_iter=1000):
    iterations = 0
    x = x0
    while abs(f(x)) > tol and iterations < max_iter:
        deriv = df(x)
        if abs(deriv) < 1e-6:
            raise ValueError("Производная близка к нулю.")
        x = x - f(x) / deriv
        iterations += 1
    return x, iterations


newton_results = []
print("\nМетод Ньютона:")
for a, b in intervals:
    x0 = (a + b) / 2.0
    try:
        root, iters = newton_method(f, df, x0, tol=1e-3)
        newton_results.append((root, iters))
        print(
            f"Начальное приближение x0 = {x0:.3f}: найденный корень x = {root:.4f} за {iters} итераций (f(x) = {f(root):.4e})")
    except Exception as e:
        print(f"Начальное приближение x0 = {x0:.3f}: {e}")

print("\nСравнение результатов (n - число итераций):")
print("{:<40}{:<30}{:<30}".format("Интервал, x0", "Метод деления отрезка / 2", "Метод Ньютона"))
for i, ((a, b), (root_bis, iters_bis)) in enumerate(zip(intervals, bisection_results)):
    x0 = (a + b) / 2.0
    if i < len(newton_results):
        root_newt, iters_newt = newton_results[i]
        print("{:<40}{:<30}{:<30}".format(
            f"[{a:.3f}, {b:.3f}], x0 = {x0:.3f}",
            f"x = {root_bis:.4f}, n = {iters_bis}",
            f"x = {root_newt:.4f}, n = {iters_newt}"
        ))
