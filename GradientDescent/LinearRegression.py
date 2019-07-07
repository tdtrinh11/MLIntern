import numpy as np 
from sklearn.linear_model import LinearRegression

np.random.seed(1)
# tao du lieu ngau nhien de test
X = np.random.rand(1000)
y = 4 + 3 * X + .5*np.random.randn(1000)

# print(X.reshape(-1,1))
model = LinearRegression()
model.fit(X.reshape(-1,1), y.reshape(-1,1))

w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b,w])
# print(sol_sklearn)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis=1)
w_init = np.array([2, 1])

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w)-y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N * np.linalg.norm(y - Xbar.dot(w))**2

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta * grad(w[-1])
        # print(w_new,'/',len(w_new))
        print((grad(w_new)))
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it) 

(w1, it1) = myGD(w_init, grad, 1)
# print(w1[-1])
print('w = {0}, it = {1}'.format(w1[-1].T, it1+1))