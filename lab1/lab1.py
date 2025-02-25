def gaussian_elimination(A, b):
    n = len(A)
    for i in range(n):
        A[i].append(b[i])

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k
        A[i], A[max_row] = A[max_row], A[i]

        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n + 1):
                A[k][j] -= factor * A[i][j]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]

    return x

def calculate_residual(A, b, x):
    n = len(A)
    residuals = [abs(b[i] - sum(A[i][j] * x[j] for j in range(n))) for i in range(n)]
    print(residuals)
    delta = max(residuals)
    relative_delta = delta / max(abs(x_i) for x_i in x)
    return delta, relative_delta

A = [[2, 1, -0.1, 1],
      [0.4, 0.5, 4, -8.5],
      [0.3, -1, 1, 5.2],
      [1, 0.2, 2.5, -1]]
b = [2.7, 21.9, -3.9, 9.9]

A1 = [[2, 1, -0.1, 1],
      [0.4, 0.5, 4, -8.5],
      [0.3, -1, 1, 5.2],
      [1, 0.2, 2.5, -1]]
b1 = [2.7, 21.9, -3.9, 9.9]

# A1 = [[2, -1, 0, 1],
#       [1, 3, -2, 2],
#       [0, 1, 2, -1],
#       [1, -2, 1, 3]]
# b1 = [5, 3, 4, 7]
#
# A2 = [[4, -2, 1, 3],
#       [-2, 4, -2, 1],
#       [1, -2, 3, -1],
#       [3, 1, -1, 2]]
# b2 = [8, -3, 4, 7]

# A3 = [[4, -2, 1, 3],
#       [8, -4, 3, 6],
#       [-2, 4, -2, 1],
#       [1, -2, 3, -1],
#       [3, 1, -1, 2]]
# b3 = [8, 16, -3, 4, 7]

solution = gaussian_elimination(A, b)
delta, rel_delta = calculate_residual(A1, b1, solution)
print("Решение системы:", solution)
print("Невязка:", delta, "Относительная невязка:", rel_delta)

# solution1 = gaussian_elimination(A1, b1)
# delta1, rel_delta1 = calculate_residual(A1, b1, solution1)
# print("Решение системы 1:", solution1)
# print("Невязка:", delta1, "Относительная невязка:", rel_delta1)
#
# solution2 = gaussian_elimination(A2, b2)
# delta2, rel_delta2 = calculate_residual(A2, b2, solution2)
# print("Решение системы 2:", solution2)
# print("Невязка:", delta2, "Относительная невязка:", rel_delta2)

# solution3 = gaussian_elimination(A3, b3)
# delta3, rel_delta3 = calculate_residual(A3, b3, solution3)
# print("Решение системы 3:", solution3)
# print("Невязка:", delta3, "Относительная невязка:", rel_delta3)