import math


def f(x):
    x1, x2 = x
    return x1 ** 4 + x2 ** 2 + x1 * x2


def grad(x):
    x1, x2 = x
    return 4 * x1 ** 3 + x2, 2 * x2 + x1


def line_search(x, g, eps):
    def fi(t):
        return f((x[0] - t * g[0], x[1] - t * g[1]))

    a, b = 0.0, 1.0
    while fi(b) < fi(b / 2):
        b *= 2
    delta = eps / 10
    while (b - a) / 2 > eps:
        t1 = (a + b - delta) / 2
        t2 = (a + b + delta) / 2
        if fi(t1) < fi(t2):
            b = t2
        else:
            a = t1
    return (a + b) / 2


def steepest_descent(x0, eps):
    x = x0
    for i in range(1, 10001):
        g = grad(x)
        if max(abs(g[0]), abs(g[1])) < eps:
            return x, f(x), i
        t = line_search(x, g, eps)
        x = (x[0] - t * g[0], x[1] - t * g[1])
    raise RuntimeError("Did not converge")


if __name__ == "__main__":
    x0 = (0.0, 1.0)
    eps = 1e-3

    x_num, f_num, iters = steepest_descent(x0, eps)
    print(f"Numeric solution:")
    print(f"  x = ({x_num[0]:.10f}, {x_num[1]:.10f})")
    print(f"  f = {f_num:.10f}")
    print(f"  iterations = {iters}")
    print()

    x1a = -1 / (2 * math.sqrt(2))
    x2a = 1 / (4 * math.sqrt(2))
    f_anal = f((x1a, x2a))
    print(f"Analytic solution:")
    print(f"  x = ({x1a:.10f}, {x2a:.10f})")
    print(f"  f = {f_anal:.10f}")
    print()

    dx1 = x_num[0] - x1a
    dx2 = x_num[1] - x2a
    df = f_num - f_anal
    print(f"Differences:")
    print(f"  Δx = ({dx1:.2e}, {dx2:.2e})")
    print(f"  Δf = {df:.2e}")
