import numpy as np
from sklearn.base import RegressorMixin

class SGDLinearRegressor(RegressorMixin):
    def __init__(self, 
                 lr=0.01, regulatization=1., 
                 delta_converged=1e-3, max_steps=1000, batch_size=64):
        self.lr = lr
        self.regularization = regulatization
        self.delta_converged = delta_converged
        self.max_steps = max_steps
        self.batch_size = batch_size
    
        self.W = None
        self.b = None
    
    def fit(self, X, Y):
        X = np.hstack((X, np.ones(X.shape[0]).reshape((-1, 1))))
        n, d = X.shape
        self.W = np.zeros(d)
        self.b = 0
        for i in range(self.max_steps):
            inds = np.random.permutation(np.arange(X.shape[0]))[:self.batch_size]
            X_batch = X[inds]
            Y_batch = Y[inds]
            grad = (X_batch.T @ (X_batch @ self.W + self.b - Y_batch )) / self.batch_size
            grad[:-1] += self.regularization * self.W[:-1]
            if np.linalg.norm(self.W - grad) < self.delta_converged:
                self.W -= self.lr * grad
                break
            self.W -= self.lr * grad
        self.b = self.W[-1]
        self.W = self.W[:-1]
    
    def predict(self, X):
        return X @ self.W + self.b