import numpy as np
np.random.seed(1)
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 5
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[0], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

print(X0.T)
print(X1.T)
X = np.concatenate((X0, X1, X2), axis = 0)
# print(X)