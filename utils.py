import numpy as np
from random import random


def rand(a, b):
    return (b - a) * random() + a


def make_matrix(m, n, fill=0.0):  # 创造一个指定大小的矩阵
    matrix = []
    for i in range(m):
        matrix.append([fill] * n)
    return matrix


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    m = sigmoid(x)
    return m * (1 - m)
