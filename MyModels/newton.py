import numpy as np


class MyNewtonLogisticRegression:
    def __init__(self):
        self.w_ = None

    def fit(self, X, y, max_iter=100, lr=0.01):
        X = np.array(X)
        y = np.array(y)
