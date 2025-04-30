import math


def f(x, Y):
    return [Y[1], math.exp(5 * x) + 4 * Y[1] - 3 * Y[0]]


def rk4_step(f, x, Y, h):
    k1 = f(x, Y)
    k1 = [h * v for v in k1]
    Y1 = [Y[i] + 0.5 * k1[i] for i in range(2)]

    k2 = f(x + 0.5 * h, Y1)
    k2 = [h * v for v in k2]
    Y2 = [Y[i] + 0.5 * k2[i] for i in range(2)]

    k3 = f(x + 0.5 * h, Y2)
    k3 = [h * v for v in k3]
    Y3 = [Y[i] + k3[i] for i in range(2)]

    k4 = f(x + h, Y3)
    k4 = [h * v for v in k4]
    return [Y[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(2)]


def exact_solution(x):
    y = (1 / 8) * math.exp(x) + (11 / 4) * math.exp(3 * x) + (1 / 8) * math.exp(5 * x)
    y_p = (1 / 8) * math.exp(x) + (33 / 4) * math.exp(3 * x) + (5 / 8) * math.exp(5 * x)
    return y, y_p


def runge_kutta_fixed(f, x0, Y0, x_end, h):
    p = 4
    results = []
    x = x0
    Y = Y0[:]
    ey, eyp = exact_solution(x)
    results.append((x, Y[0], ey, Y[1], eyp, 0.0, 0.0, h))
    while x < x_end:
        h_step = min(h, x_end - x)

        Y_h = rk4_step(f, x, Y, h_step)
        Y_half = rk4_step(f, x, Y, h_step / 2)
        Y_h2 = rk4_step(f, x + h_step / 2, Y_half, h_step / 2)

        err_local = abs(Y_h[0] - Y_h2[0]) / (2 ** p - 1)
        Y_corr = [Y_h[i] + (Y_h[i] - Y_h2[i]) / (2 ** p - 1) for i in range(2)]

        x += h_step
        Y = Y_corr
        ey, eyp = exact_solution(x)
        err_global = abs(Y[0] - ey)
        results.append((x, Y[0], ey, Y[1], eyp, err_local, err_global, h_step))
    return results


def runge_kutta_adaptive(f, x0, Y0, x_end, h0, eps):
    results = []
    x = x0
    Y = Y0[:]
    h = h0
    ey, eyp = exact_solution(x)
    results.append((x, Y[0], ey, Y[1], eyp, 0.0, 0.0, h))

    while x < x_end:
        if x + h > x_end:
            h = x_end - x

        Y_h = rk4_step(f, x, Y, h)

        Y_half = rk4_step(f, x, Y, h / 2)
        Y_h2 = rk4_step(f, x + h / 2, Y_half, h / 2)

        err_local = abs(Y_h[0] - Y_h2[0]) / 15.0

        Y_corr = [Y_h[i] + (Y_h[i] - Y_h2[i]) / 15.0 for i in range(2)]
        err_global = abs(Y_corr[0] - ey)

        h_opt = h * (eps / err_local) ** (1.0 / 5.0)
        h_new = 0.9 * h_opt

        if err_local <= eps:
            x += h
            Y = Y_corr
            ey, eyp = exact_solution(x)
            results.append((x, Y[0], ey, Y[1], eyp, err_local, err_global, h))
            h = h_new
        else:
            h = h_new

    return results


def print_results(results, title):
    print(f"\n--- {title} ---")
    print("{:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8}".format(
        "x", "Approx y", "Exact y", "Approx y'", "Exact y'", "Local Err", "Global Err", "h"))
    for x_val, ay, ey, apyp, ey_p, el, eg, hs in results:
        print(f"{x_val:8.4f} {ay:12.6f} {ey:12.6f} {apyp:12.6f} {ey_p:12.6f} {el:12.2e} {eg:12.2e} {hs:8.4f}")


def main():
    x0, x_end = 0.0, 1.0
    Y0 = [3.0, 9.0]
    h = 0.03125
    res_fixed = runge_kutta_fixed(f, x0, Y0, x_end, h)
    print_results(res_fixed, "Runge-Kutt fixed")
    eps = 1e-3
    res_adapt = runge_kutta_adaptive(f, x0, Y0, x_end, h, eps)
    print_results(res_adapt, f"Runge-Kutt adaptive")


if __name__ == "__main__":
    main()
