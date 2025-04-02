import math


def f(x):
    return 0.25 * x * math.exp((x ** 2) / 2)


def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return h * s


def simpson_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Для метода Симпсона нужно чётное n.")
    h = (b - a) / n
    s = f(a) + f(b)
    s_odd = 0.0
    s_even = 0.0

    for k in range(1, n):
        xk = a + k * h
        if k % 2 == 1:
            s_odd += f(xk)
        else:
            s_even += f(xk)

    return (h / 3) * (s + 4 * s_odd + 2 * s_even)


def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        mid = a + (i + 0.5) * h
        s += f(mid)
    return h * s


def runge_integration(method, f, a, b, eps, p, start_n=2):
    n = start_n
    I_n = method(f, a, b, n)

    while True:
        I_2n = method(f, a, b, 2 * n)
        R = (I_2n - I_n) / (2 ** p - 1)

        if abs(R) < eps:
            I_extrap = I_2n + R
            return I_2n, R, I_extrap, n

        n *= 2
        I_n = I_2n


if __name__ == "__main__":
    a = 0
    b = 2
    eps = 0.001

    integral = (math.exp(2) - 1) / 4


    integral_star_x2_trap, R_trap, I_extrap_trap, n_trap = runge_integration(trapezoidal_rule, f, a, b, eps, p=2)
    integral_star_x2_simpson, R_simpson, I_extrap_simpson, n_simpson = runge_integration(simpson_rule, f, a, b, eps, p=4)
    integral_star_x2_midpoint, R_midpoint, I_extrap_midpoint, n_midpoint = runge_integration(midpoint_rule, f, a, b, eps, p=2)

    print("Eps: ", eps)
    print("I: ", integral)
    print("{:<13}{:<25}{:<25}{}".format("", "Метод трапеций:", "Метод Симпсона:", "Метод центр.прямоуг.:"))
    print("{:<13}{:<25}{:<25}{}".format("n:", str(n_trap), str(n_simpson), str(n_midpoint)))
    print("{:<13}{:<25}{:<25}{}".format("I^*_h/2:", str(integral_star_x2_trap), str(integral_star_x2_simpson),
                                        str(integral_star_x2_midpoint)))
    print("{:<13}{:<25}{:<25}{}".format("R:", str(R_trap), str(R_simpson), str(R_midpoint)))
    print("{:<13}{:<25}{:<25}{}".format("I^*_h/2 + R:", str(I_extrap_trap), str(I_extrap_simpson),
                                        str(I_extrap_midpoint)))
