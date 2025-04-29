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


def runge_kutta_fixed_with_richardson(f, x0, Y0, x_end, h):
    p = 4
    results = []
    x = x0
    Y = Y0[:]
    ey, eyp = exact_solution(x)
    results.append((x, Y[0], ey, Y[1], eyp, 0.0, h))
    while x < x_end:
        h_step = min(h, x_end - x)
        # standard RK4 with step h
        Y_h = rk4_step(f, x, Y, h_step)
        # two half-steps RK4 for Richardson
        Y_half = rk4_step(f, x, Y, h_step / 2)
        Y_h2 = rk4_step(f, x + h_step / 2, Y_half, h_step / 2)
        Y_corr = [Y_h[i] + (Y_h[i] - Y_h2[i]) / (2 ** p - 1) for i in range(2)]
        x += h_step
        Y = Y_corr
        ey, eyp = exact_solution(x)
        err = abs(Y[0] - ey)
        results.append((x, Y[0], ey, Y[1], eyp, err, h_step))
    return results


def runge_kutta_adaptive_with_richardson(f, x0, Y0, x_end, h0, eps):
    p = 4
    max_factor = 5.0
    min_factor = 0.1
    results = []
    x = x0
    Y = Y0[:]
    h = h0
    ey, eyp = exact_solution(x)
    results.append((x, Y[0], ey, Y[1], eyp, 0.0, h))
    while x < x_end:
        # ensure we can take two half-steps
        if x + 2 * h > x_end:
            h = (x_end - x) / 2.0
        # full step
        Y_h = rk4_step(f, x, Y, h)
        # two half-steps
        Y_half = rk4_step(f, x, Y, h / 2)
        Y_h2 = rk4_step(f, x + h / 2, Y_half, h / 2)
        # Richardson correction
        Y_corr = [Y_h[i] + (Y_h[i] - Y_h2[i]) / (2 ** p - 1) for i in range(2)]
        # error estimate
        err = abs(Y_h[0] - Y_h2[0]) / (2 ** p - 1)
        # adapt step
        h_opt = h * (eps / err) ** (1.0 / (p + 1)) if err > 0 else max_factor * h
        h_new = 0.9 * h_opt
        h_new = max(min_factor * h, min(max_factor * h, h_new))
        # accept or reject
        if err <= eps:
            # advance
            x += h
            Y = Y_corr
            ey, eyp = exact_solution(x)
            # record
            results.append((x, Y[0], ey, Y[1], eyp, err, h))
            h = h_new
        else:
            h = h_new
    return results


def print_results(results):
    print("{:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8}".format(
        "x", "Approx y", "Exact y", "Approx y'", "Exact y'", "Error", "h"))
    for x_val, ay, ey, apyp, ey_p, err, h in results:
        print(f"{x_val:8.4f} {ay:12.6f} {ey:12.6f} {apyp:12.6f} {ey_p:12.6f} {err:12.2e} {h:8.4f}")


def main():
    x0, x_end = 0.0, 1.0
    Y0 = [3.0, 9.0]
    h_fixed = 0.03125
    res_fixed = runge_kutta_fixed_with_richardson(f, x0, Y0, x_end, h_fixed)
    print("--- Fixed-step RK4 with Richardson extrapolation ---")
    print_results(res_fixed)
    # adaptive-step
    h0 = 0.5
    eps = 0.001
    res_adapt = runge_kutta_adaptive_with_richardson(f, x0, Y0, x_end, h0, eps)
    print("--- Adaptive-step RK4 with Richardson and eps=", eps, "---")
    print_results(res_adapt)


if __name__ == "__main__":
    main()
