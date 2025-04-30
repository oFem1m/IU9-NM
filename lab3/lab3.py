import math


def f(x, Y):
    return [Y[1], math.exp(5 * x) + 4 * Y[1] - 3 * Y[0]]


def rk4_step(func, x, Y, h):
    k1 = func(x, Y)
    k1 = [h * v for v in k1]
    Y1 = [Y[i] + 0.5 * k1[i] for i in range(2)]
    k2 = func(x + 0.5 * h, Y1)
    k2 = [h * v for v in k2]
    Y2 = [Y[i] + 0.5 * k2[i] for i in range(2)]
    k3 = func(x + 0.5 * h, Y2)
    k3 = [h * v for v in k3]
    Y3 = [Y[i] + k3[i] for i in range(2)]
    k4 = func(x + h, Y3)
    k4 = [h * v for v in k4]
    return [Y[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(2)]


def exact_solution(x):
    y = 0.125 * math.exp(x) + 2.75 * math.exp(3 * x) + 0.125 * math.exp(5 * x)
    yp = 0.125 * math.exp(x) + 8.25 * math.exp(3 * x) + 0.625 * math.exp(5 * x)
    return y, yp


def err_norm(dy, y, atol, rtol):
    tol0 = atol + rtol * abs(y[0])
    tol1 = atol + rtol * abs(y[1])
    return max(abs(dy[0]) / tol0, abs(dy[1]) / tol1)


def runge_kutta_fixed(func, x0, Y0, x_end, h_init, atol, rtol):
    h = h_init
    while True:
        results = []
        x = x0
        Y = Y0[:]
        ey, eyp = exact_solution(x)
        results.append((x, Y[0], ey, Y[1], eyp, 0.0, 0.0, h))
        ok = True
        while x < x_end:
            h_step = min(h, x_end - x)
            Y_h = rk4_step(func, x, Y, h_step)
            Y_half = rk4_step(func, x, Y, h_step / 2)
            Y_h2 = rk4_step(func, x + h_step / 2, Y_half, h_step / 2)

            dy = [Y_h2[i] - Y_h[i] for i in range(2)]

            err_local = err_norm(dy, Y_h2, atol, rtol) / 15.0
            if err_local > 1:
                ok = False
                break
            Y = [Y_h[i] + dy[i] / 15.0 for i in range(2)]
            x += h_step
            ey, eyp = exact_solution(x)
            err_global = abs(Y[0] - ey)
            results.append((x, Y[0], ey, Y[1], eyp, err_local, err_global, h_step))
        if ok:
            return results, h
        h /= 2.0


def runge_kutta_adaptive(func, x0, Y0, x_end, h0, atol, rtol):
    h_max = 0.0625
    results = []
    x = x0
    Y = Y0[:]
    h = h0
    ey, eyp = exact_solution(x)
    results.append((x, Y[0], ey, Y[1], eyp, 0.0, 0.0, h))

    while x < x_end:
        if x + h > x_end:
            h = x_end - x
        Y_h = rk4_step(func, x, Y, h)

        Y_half = rk4_step(func, x, Y, h / 2)

        Y_h2 = rk4_step(func, x + h / 2, Y_half, h / 2)
        dy = [Y_h2[i] - Y_h[i] for i in range(2)]
        err_local = err_norm(dy, Y_h2, atol, rtol) / 15.0

        Y_corr = [Y_h[i] + dy[i] / 15.0 for i in range(2)]
        if err_local == 0:
            h_opt = 2 * h
        else:
            h_opt = h * (1 / err_local) ** 0.2
        h_new = 0.9 * h_opt
        h_new = max(0.5 * h, min(2.0 * h, h_new))
        h_new = min(h_new, h_max)
        if err_local <= 1:
            x += h
            Y = Y_corr
            ey, eyp = exact_solution(x)
            err_global = abs(Y[0] - ey)
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
    h_init = 0.5
    atol = 1e-6
    rtol = 1e-3
    res_fixed, h_final = runge_kutta_fixed(f, x0, Y0, x_end, h_init, atol, rtol)
    print_results(res_fixed, "Runge–Kutta fixed")
    res_adapt = runge_kutta_adaptive(f, x0, Y0, x_end, h_init, atol, rtol)
    print_results(res_adapt, "Runge–Kutta adaptive")


if __name__ == "__main__":
    main()
