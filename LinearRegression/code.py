import numpy as np
from sklearn.linear_model import LinearRegression

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# print(X.T,"\n",y.T)
# create one columm
one = np.ones((X.shape[0],1))
# print(one.T)
# create matrix A
Xbar = np.concatenate((one, X), axis=1)
# print(Xbar.T)
# calculating
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print(f"Nghiem tu giai: {w.T}")

# cai dung ham co san
regr = LinearRegression(fit_intercept=False).fit(Xbar, y)
# fit_intercept = False for calculating the bias

# Compare two results
print('Solution found by scikit-learn  : ', regr.coef_ )
# test
X_test = np.array([[155, 160]])
Xbar_test = np.concatenate(([[1,1]], X_test), axis=0)
y_test = np.dot(w.T, Xbar_test)
print(f"Test: {y_test}")
