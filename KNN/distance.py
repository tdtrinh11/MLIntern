from __future__ import print_function
import numpy as np
from time import time
d, N = 1000, 10000 # d la so chieu, N la so diem du lieu
# tao ra tap diem ngau nhien
X = np.random.randn(N, d) # tap diem co N diem, d-chieu
z = np.random.randn(d)  # vecto d-chieu
# print(z)

# ham tinh bp khoang cach 2 diem theo cach truc tiep dung l2 norm
def dist_pp(z, x):
    d = z - x.reshape(z.shape)
    return np.sum(d*d)

# ham tinh khoang cach giua z voi moi diem trong X
def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range(N):
        res[0][i] = dist_pp(z, X[i])
    return res

# ham tinh khoang cach giua z voi moi diem trong X bang cong thuc bien doi
def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1)
    z2 = np.sum(z*z)
    return X2 + z2 - 2*X.dot(z)

# t1 = time()
# D1 = dist_ps_naive(z, X)
# print('naive point2set, running time:', time() - t1, 's')
# t1 = time()
# D2 = dist_ps_fast(z, X)
# print('fast point2set , running time:', time() - t1, 's')
# print('Result difference:', np.linalg.norm(D1 - D2))

M = 100
Z = np.random.randn(M,d)

# tinh khong cach giua moi ptu cua Z den moi phan tu cua X
# theo cac dung vong lap, moi vong lap dung ham fast
def dist_ss_0(Z, X):
    M = Z.shape[0]
    N = X.shape[0]
    res = np.zeros((M, N))
    for i in range(M):
        res[i] = dist_ps_fast(Z[i], X)
    return res

# dung cong thuc
def dist_ss_fast(Z, X):
    X2 = np.sum(X*X, 1)
    Z2 = np.sum(Z*Z, 1)
    return X2.reshape(-1, 1) +  Z2.reshape(-1, 1) - 2 * Z.dot(X.T)

t1 = time()
D3 = dist_ss_0(Z, X)
print('half fast set2set running time:', time() - t1, 's')
t1 = time()
D4 = dist_ss_fast(Z, X)
print('fast set2set running time', time() - t1, 's')
print('Result difference:', np.linalg.norm(D3 - D4))