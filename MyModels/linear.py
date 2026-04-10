import numpy as np


class MyLinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        assert len(X.shape) == 2 and len(y.shape) == 1
        assert X.shape[0] == y.shape[0]

        y = y[:, np.newaxis]
        l, n = X.shape

        X_train = np.hstack((X, np.ones((l, 1))))

        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y
        return self

    def predict(self, X):
        X = np.array(X)
        l, n = X.shape
        X_pred = np.hstack((X, np.ones((l, 1))))
        y_pred = X_pred @ self.w
        return y_pred

    def get_weights(self):
        return self.w.copy()
