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
    y_prime = ((1 / 8) * math.exp(x) + (33 / 4) * math.exp(3 * x) + (5 / 8) * math.exp(5 * x))
    return y, y_prime


def runge_kutt(f, x0, Y0, x_end, h):
    results = []
    x = x0
    Y = Y0[:]

    exact_y, exact_y_prime = exact_solution(x)
    results.append((x, Y[0], Y[1], exact_y, exact_y_prime, 0.0))

    while x < x_end:
        if x + 2*h > x_end:
            h = (x_end - x) / 2

        Y_h1 = rk_step(f, x, Y, h)
        Y_h2 = rk_step(f, x + h, Y_h1, h)

        Y_2h = rk_step(f, x, Y, 2*h)

        err = abs(Y_h2[0] - Y_2h[0]) / 15

        x += 2*h
        Y = Y_h2

        exact_y, exact_y_prime = exact_solution(x)
        results.append((x, Y[0], Y[1], exact_y, exact_y_prime, err))

    return results


def main():
    x0 = 0
    x_end = 1
    Y0 = [3, 9]
    h = 0.01

    results = runge_kutt(f, x0, Y0, x_end, h)

    print("{:>8} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
        "x", "Approx y", "Exact y", "Approx y'", "Exact y'", "Error"))
    for x_val, approx_y, exact_y, approx_yprime, exact_yprime, err in results:
        print("{:8.4f} {:14.6f} {:14.6f} {:14.6f} {:14.6f} {:14.6f}".format(
            x_val, approx_y, exact_y, approx_yprime, exact_yprime, err))


if __name__ == "__main__":
    main()
