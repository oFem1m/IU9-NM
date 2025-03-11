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


A = [[2, -1, 1],
     [1, 3, 2],
     [1, -1, 2]]
b = [8, 13, 3]

print("Решение:", gaussian_elimination(A, b))
