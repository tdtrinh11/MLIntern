import numpy as np
import matplotlib.pyplot as plt

# doc du lieu tu file
f = open('../Data/ex1data2.txt', 'r')
x = f.readline()
X1 = []
X2 = []
y = []
while x != "":
    z = x.split(',')
    X1.append(float(z[0]))
    X2.append(float(z[1]))
    y.append(float(z[2]))
    
    x = f.readline()

# print(len(X1), len(X2), len(y))
# print(np.array([X1]))
one = np.ones((len(X1), 1))
# print(one)
X = np.concatenate((one, np.array([X1]).T, np.array([X2]).T), axis=1)
# print(X)
y = np.array([y])