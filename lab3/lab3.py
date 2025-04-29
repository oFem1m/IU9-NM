import math


def f(x, Y):
    return [Y[1], math.exp(5 * x) + 4 * Y[1] - 3 * Y[0]]


def rk_step(f, x, Y, h):
    k1 = [h * val for val in f(x, Y)]
    Y1 = [Y[i] + 0.5 * k1[i] for i in range(len(Y))]

    k2 = [h * val for val in f(x + 0.5 * h, Y1)]
    Y2 = [Y[i] + 0.5 * k2[i] for i in range(len(Y))]

    k3 = [h * val for val in f(x + 0.5 * h, Y2)]
    Y3 = [Y[i] + k3[i] for i in range(len(Y))]

    k4 = [h * val for val in f(x + h, Y3)]

    Y_next = [Y[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(len(Y))]
    return Y_next


def exact_solution(x):
    y = (1 / 8) * math.exp(x) + (11 / 4) * math.exp(3 * x) + (1 / 8) * math.exp(5 * x)
    y_prime = (1 / 8) * math.exp(x) + (33 / 4) * math.exp(3 * x) + (5 / 8) * math.exp(5 * x)
    return y, y_prime


def runge_kutt(f, x0, Y0, x_end, h):
    results = []
    x = x0
    Y = Y0[:]

    exact_y, exact_y_prime = exact_solution(x)
    results.append((x, Y[0], Y[1], exact_y, exact_y_prime, 0.0, h))

    while x < x_end:
        if x + 2 * h > x_end:
            break

        Y_h_prev = rk_step(f, x, Y, h)
        Y_h = rk_step(f, x + h, Y_h_prev, h)

        Y_2h = rk_step(f, x, Y, 2 * h)

        err = abs(Y_h[0] - Y_2h[0]) / 15.0

        x += 2 * h
        Y = Y_h

        exact_y, exact_y_prime = exact_solution(x)
        results.append((x, Y[0], Y[1], exact_y, exact_y_prime, err, h))

    return results


def runge_kutt_adaptive(f, x0, Y0, x_end, h0, eps):
    results = []
    x = x0
    Y = Y0[:]
    h = h0

    ey, ey_p = exact_solution(x)
    results.append((x, Y[0], Y[1], ey, ey_p, 0.0, h))

    while x < x_end:
        if x + h > x_end:
            h = x_end - x

        Y_mid = rk_step(f, x, Y, h)
        Y_h = rk_step(f, x + h, Y_mid, h)

        Y_2h = rk_step(f, x, Y, 2 * h)

        err = abs(Y_h[0] - Y_2h[0]) / 15.0

        h_opt = h * (eps / err) ** (1.0 / 5.0) if err != 0 else 2 * h
        h_new = 0.9 * h_opt

        if err <= eps:
            Y = [Y_h[i] + (Y_h[i] - Y_2h[i]) / 15.0 for i in range(len(Y))]
            x += h
            ey, ey_p = exact_solution(x)
            results.append((x, Y[0], Y[1], ey, ey_p, err, h))
            h = h_new
        else:
            h = h_new

    return results


def print_results(results, title):
    print(f"\n--- {title} ---")
    print("{:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8}".format(
        "x", "Approx y", "Exact y", "Approx y'", "Exact y'", "Error", "h"))
    for row in results:
        x_val, ay, ay_p, ey, ey_p, err, h = row
        print("{:8.4f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.2e} {:8.4f}".format(
            x_val, ay, ey, ay_p, ey_p, err, h))


def main():
    x0 = 0.0
    x_end = 1.0
    Y0 = [3.0, 9.0]
    h = 0.065
    eps = 0.001

    res_const = runge_kutt(f, x0, Y0, x_end, h)
    print_results(res_const, "Constant-step RK4")
    res_adapt = runge_kutt_adaptive(f, x0, Y0, x_end, h, eps)
    print_results(res_adapt, f"Adaptive RK4 (eps={eps})")


if __name__ == "__main__":
    main()
