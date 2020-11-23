import numpy as np
import math
import matplotlib.pyplot as plt

tx = 100
tc = 2
tu = 5

def lambdaK (k):
    return 1.0 / tx

def r(k):
    return 1.0 / (tc + (k/2.0) * tu)

def p(s, n):
    if (s == 0):
        acc = 0
        for k in range(n):
            temp = 1
            for i in range(k):
                temp *= (lambdaK(i) / (r(i + 1)*1.0))
            acc += temp
        return 1.0 / (1 + acc)
    else:
        acc = p(0, n)
        for k in range(s):
            acc *= (lambdaK(k) / (r(k + 1)*1.0))
    return acc

def N(n):
    acc = 0
    for i in range(n + 1):
        acc += i * p(i, n)
    return acc

def R(n):
    return N(n) / lambdaK(n)


def Wq(n):
    acc = 0;
    for i in range(n + 1):
        acc += r(i) * p(i, n)
    return R(n) - acc

def plot(n):
    n_N = np.empty(n)
    Wqs = np.empty(n)
    for i in range(n):
        I = i + 1
        n_N = np.append(n_N, I - N(I))
        Wqs = np.append(Wqs, Wq(I))
    plt.plot(Wqs, n_N)
    plt.xlabel("WQ")
    plt.ylabel("n - N")
    plt.show()

plot(32)
