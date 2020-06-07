import numpy as np
from support import complex_pow


def v_inv_ldu(alpha, n):
    u = build_u_matrices(alpha, n)
    u_prod = multiply_u_matrices(u)
    dl = build_dl_matrices(alpha, n)
    dl_prod = multiply_dl_matrices(dl)
    return u_prod @ dl_prod


def build_u_matrices(alpha, n):
    u = np.zeros((n-1, n, n), dtype=np.complex_)
    for k in range(n-1):
        for j in range(n):
            u[k, j, j] = 1
            if j > k-1 and j < n-1:
                u[k, j, j+1] = -complex_pow(alpha, k)
    return u


def build_dl_matrices(alpha, n):
    d = np.zeros((n - 1, n, n), dtype=np.complex_)
    for k in range(n - 1):
        for i in range(n):
            if i <= k:
                d[k, i, i] = 1
            if i > k:
                d[k, i, i] = 1 / (complex_pow(alpha, i) - complex_pow(alpha, k))
    l = np.zeros((n - 1, n, n), dtype=np.complex_)
    for k in range(n - 1):
        for i in range(n):
            l[k, i, i] = 1
            if i > k:
                l[k, i, k] = -1
    dl = np.zeros((n - 1, n, n), dtype=np.complex_)
    for i in range(n-1):
        dl[i, :, :] = d[i, :, :] @ l[i, :, :]
    return dl


def multiply_u_matrices(a):
    n = a.shape[1]
    out = np.zeros((n, n))
    out = a[0, :, :]
    for i in range(1, n - 1):
        out = out @ a[i, :, :]
    return out


def multiply_dl_matrices(a):
    n = a.shape[1]
    out = np.zeros((n, n))
    out = a[0, :, :]
    for i in range(1, n - 1):
        out = a[i, :, :] @ out
    return out
