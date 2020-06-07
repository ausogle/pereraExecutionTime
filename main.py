import numpy as np
import cmath
from support import vandermonde, gauss, vand_and_b
from datetime import datetime as dt
from algorithm import v_inv_ldu
from from_matlab import x_mtlb
import matplotlib.pyplot as plt

n = np.arange(64, 124, 4)

time = np.zeros((n.shape[0], 2))
norm_diff = np.zeros((n.shape[0], 1))

alpha = cmath.exp(1j*np.pi/18)

ex_lu = []
ex_alg = []
ex_matlab = []

for i in range(n.shape[0]):
    b = np.random.rand(n[i], 1)
    v = vandermonde(alpha, n[i])

    a = vand_and_b(alpha, n[i], b)
    start = dt.now()                        # LAPACK SOLVER
    x_lu = gauss(a)
    end = dt.now()
    ex_lu.append((end - start))
    residual = v@x_lu - b

    # start = dt.now()                        # LDU solver
    # vi = v_inv_ldu(alpha, n[i])
    # x_alg = vi @ b
    # end = dt.now()
    # ex_alg.append((end - start))
    # residual2 = v@x_alg - b

    # start = dt.now()                        # Column-wise algorithm
    # x_matlab = x_mtlb(n[i], b, t=-np.pi / 18)
    # end = dt.now()
    # ex_matlab.append((end - start))
    # residual3 = v @ x_matlab-b

    print(n[i])
    # print(np.linalg.norm(residual))
    # print(np.linalg.norm(residual2))
    # print(np.linalg.norm(residual3))

print("n")
print(n)
print("Gaussian")
print(ex_lu)
# print("Matrix multiplication")
# print(ex_alg)
# print("Vector ")
# print(ex_matlab)
# fig = plt.figure()
# ax = fig.gca()
# plt.plot(n, ex_lu, label="LAPACK Gaussian Elim")
# plt.plot(n, ex_lu, label="Gaussian Elim")
# plt.plot(n, ex_alg, label="Matrix")
# plt.plot(n, ex_matlab, label="Vector")
# ax.legend()
# plt.show()
