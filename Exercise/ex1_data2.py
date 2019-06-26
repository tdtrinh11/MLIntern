import numpy as np
import matplotlib.pyplot as plt

# doc du lieu tu file
f = open('ex1data2.txt', 'r')
x = f.readline()
X = [[]]
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