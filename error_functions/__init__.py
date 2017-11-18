import numpy as np

__all__ = ["MSE"]

def MSE(x, y):
    d = x - y
    return np.sqrt(np.dot(d, d)) / len(d)