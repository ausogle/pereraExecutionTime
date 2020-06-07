import numpy as np


def x_mtlb(n, b, t=np.pi / 8):
    m = 2*n
    x = np.zeros((n, 1), dtype=np.complex_)
    A = np.zeros((n, m), dtype=np.complex_)

    for k in range(0, n):
        x[k] = np.cos(k*t)-1j*np.sin(k*t)
        A[k, 0] = b[k]

    for k in range(n-1):
        for j in range(n):
            if j > k:
                A[j, k+1] = 1/(x[j] - x[k]) * (A[j, k] - A[k, k])
            else:
                A[j, k+1] = A[j, k]

    # matches up to here
    for k in range(n-1, m-1):
        for j in range(n):
            if j < (m-k-2) or j+1 == n:
                A[j, k+1] = A[j, k]
            else:
                A[j, k+1] = A[j, k] - (A[j+1, k] * x[m-k-2])

    return A[:, m-1]

