import numpy as np
from scipy.linalg.lapack import cgesv


def vandermonde(alpha, n):
    v = np.zeros((n, n), dtype=np.complex_)
    for i in range(n):
        for j in range(n):

            v[i, j] = complex_pow(alpha, i * j)
    return v


def vand_and_b(alpha, n, b):
    v = np.zeros((n, n+1), dtype=np.complex_)
    for i in range(n):
        for j in range(n):
            v[i, j] = complex_pow(alpha, i * j)
    for i in range(n):
        v[i, n] = b[i]
    return v


def complex_pow(a, k):
    if k == 0:
        return 1
    prod = a
    for i in range(k-1):
        prod = prod * a
    return prod


def lapack_solver(a, b):
    c = cgesv(a, b)[2]
    # residual2 = a@c-b
    return c


def gauss(a):
    n = a.shape[0]
    for i in range(0, n):
        maxEl = abs(a[i][i])
        maxRow = i
        for k in range(i + 1, n):
            if abs(a[k][i]) > maxEl:
                maxEl = abs(a[k][i])
                maxRow = k

        for k in range(i, n + 1):
            tmp = a[maxRow][k]
            a[maxRow][k] = a[i][k]
            a[i][k] = tmp

        for k in range(i + 1, n):
            c = -a[k][i]/a[i][i]
            for j in range(i, n + 1):
                if i == j:
                    a[k][j] = 0
                else:
                    a[k][j] += c * a[i][j]
    x = np.zeros((n, 1), dtype=np.complex_)
    for i in range(n-1, -1, -1):
        x[i] = a[i][n] / a[i][i]
        for k in range(i-1, -1, -1):
            a[k][n] -= a[k][i] * x[i]

    return x
