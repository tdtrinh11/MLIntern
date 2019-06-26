import numpy as np
import matplotlib.pyplot as plt

# doc du lieu tu file
f = open('ex1data1.txt', 'r')
x = f.readline()
X = []
y = []
while x != "":
    z = x.split(',')
    for i in range(len(z)):
        z[i] = z[i].split()
    # print(y[1][0])
    X.append(float(z[0][0]))
    y.append(float(z[1][0]))
    # print(y)
    x = f.readline()

X = np.array([X]).T
y = np.array([y]).T
# print(X)

# X la du lieu cot dau tien
# y la du lieu cot thu 2

# thuat toan hoi quy
# tao cot 1
one = np.ones((X.shape[0], 1))
# print(one)
# X.shape[0] tra ve so hang cua ma tran X

# ma tran dung de tinh toan
Xbar = np.concatenate((one, X), axis = 1)
# print(Xbar)

# tinh toan
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
# print('w = ', w)
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(5, 15)
y0 = w_0 + w_1 * x0
# print('x0 =', x0)
# print(y0)

# ve do thi
plt.plot(X.T, y.T, 'ro')
plt.plot(x0, y0)
plt.axis([5, 15, -5, 20])
plt.xlabel("abc")
plt.ylabel("xyz")
plt.show()