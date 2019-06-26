import numpy as np
import math
import matplotlib.pyplot as plt 
import time

def grad(x):
    return (2*x+5*np.cos(x))

def cost(x):
    return (x**2+5*np.sin(x))

def myGD1(eta, x0):
    start = time.time()
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    ti = time.time()-start
    return (x, it, ti)

(x1, it1, ti1) = myGD1(.1, -5)
(x2, it2, ti2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1), "in time =", round(ti1, 7))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2), "in time =", round(ti2, 7))