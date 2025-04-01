import numpy as np

# из лабы 0, адаптированный под numpy
def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]

    aug = np.hstack((A, b.reshape(-1, 1)))

    for i in range(n):
        pivot_row = i + np.argmax(np.abs(aug[i:, i]))
        aug[[i, pivot_row]] = aug[[pivot_row, i]]

        for k in range(i + 1, n):
            factor = aug[k, i] / aug[i, i]
            aug[k, i:] -= factor * aug[i, i:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = aug[i, n] / aug[i, i]
        aug[:i, n] -= aug[:i, i] * x[i]

    return x


def f(x):
    return 0.25 * x * np.exp(x ** 2 / 2)


def build_cubic_spline_coeffs(x, y):
    n = len(x) - 1

    a = np.copy(y)
    b = np.zeros(n)
    c = np.zeros(n + 1)
    d = np.zeros(n)

    h = np.diff(x)

    A = np.zeros((n - 1, n - 1))
    rhs = np.zeros(n - 1)

    for i in range(1, n):
        index = i - 1

        if index > 0:
            A[index, index - 1] = h[i - 1]
        A[index, index] = 2.0 * (h[i - 1] + h[i] if i < n else h[i - 1] + h[i - 1])
        if index < n - 2:
            A[index, index + 1] = h[i]

        rhs[index] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c_internal = gaussian_elimination(A, rhs)

    c[1:n] = c_internal

    for i in range(n - 1):
        b[i] = ((y[i + 1] - y[i]) / h[i] - (h[i] / 3.0) * (2.0 * c[i] + c[i + 1]))
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    b[n - 1] = ((y[n] - y[n - 1]) / h[n - 1] - (2.0 * c[n - 1] * h[n - 1] / 3.0))
    d[n - 1] = - (c[n] / (3.0 * h[n - 1]))

    return a, b, c, d


def spline_value(x_val, x_nodes, a, b, c, d):
    n = len(x_nodes) - 1
    if x_val <= x_nodes[0]:
        i = 0
    elif x_val >= x_nodes[n]:
        i = n - 1
    else:
        i = np.searchsorted(x_nodes, x_val) - 1

    dx = x_val - x_nodes[i]
    return a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)


def main():
    a_val = 0.0
    b_val = 2.0
    n = 32
    x_nodes = np.linspace(a_val, b_val, n + 1)
    y_nodes = f(x_nodes)
    h = (b_val - a_val) / n

    print("Node table (x_i, y_i):")
    for i in range(n + 1):
        print(f"i = {i:2d}, x = {x_nodes[i]:.5f}, y = {y_nodes[i]:.5f}")
    print()

    A, B, C, D = build_cubic_spline_coeffs(x_nodes, y_nodes)

    print("Spline coefficients by segments [x_i, x_{i+1}]:")
    print(" i      a[i]          b[i]          c[i]          d[i]")
    for i in range(n):
        print(f"{i:2d}  {A[i]:12.6f}  {B[i]:12.6f}  {C[i]:12.6f}  {D[i]:12.6f}")
    print(f"c[n] = {C[n]:.6f}\n")
    print("Comparison of f(x*) and S(x*) at the midpoints of the segments:")
    print(" i     x*          f(x*)       S(x*)")
    for i in range(1, n + 1):
        x_star = a_val + (i - 0.5) * h
        f_star = f(x_star)
        s_star = spline_value(x_star, x_nodes, A, B, C, D)
        print(f"{i:2d}  {x_star:10.5f}  {f_star:10.5f}  {s_star:10.5f}")
    print()

    x_user_str = input("Enter an x: ")
    try:
        x_user = float(x_user_str)
        f_user = f(x_user)
        s_user = spline_value(x_user, x_nodes, A, B, C, D)
        print(f"\nIn point x = {x_user:.5f}:")
        print(f"f(x) = {f_user:.6f}")
        print(f"S(x) = {s_user:.6f}")
    except ValueError:
        print("Error.")


if __name__ == "__main__":
    main()
