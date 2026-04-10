import numpy as np


class MyLinearRegression:
    def __init__(self):
        self.w_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        assert len(X.shape) == 2 and len(y.shape) == 1
        assert X.shape[0] == y.shape[0]

        y = y[:, np.newaxis]
        l, n = X.shape

        X_train = np.hstack((X, np.ones((l, 1))))

        self.w_ = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y
        return self

    def predict(self, X):
        X = np.array(X)
        l, n = X.shape
        X_pred = np.hstack((X, np.ones((l, 1))))
        y_pred = X_pred @ self.w_
        return y_pred

    def get_weights(self):
        return self.w_.copy()


class MyLogisticRegression:
    @staticmethod
    def sigmoid(t):
        return 1. / (1 + np.exp(-t))

    @staticmethod
    def grad(X, y, y_pred):
        l = X.shape[0]
        grad = (X.T @ (y_pred - y)) / l

        return grad

    def __init__(self):
        self.w_ = None
        self.intercept_ = None

    def fit(self, X, y, max_iter=100, lr=0.01):
        X = np.array(X)
        y = np.array(y)
        assert len(X.shape) == 2 and len(y.shape) == 1
        assert X.shape[0] == y.shape[0]

        y = y[:, np.newaxis]
        l, n = X.shape
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        weights = np.random.randn(n + 1, 1)

        losses = []
        for iter_num in range(max_iter):
            logits = X @ weights
            y_pred = self.sigmoid(logits)
            grad = self.grad(X, y, y_pred)

            weights -= lr * grad

            loss = self.loss(y, y_pred)
            losses.append(loss)
        self.w_ = weights[1:]
        self.intercept_ = weights[0]

        return losses

    def predict_proba(self, X):
        X = np.array(X)
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        weights = np.concatenate([self.intercept_.reshape(-1, 1), self.w_], axis=0)
        logits = (X @ weights)

        return self.sigmoid(logits)

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
