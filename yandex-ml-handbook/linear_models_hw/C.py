import numpy as np
import sklearn
from sklearn.linear_model import Ridge

class ExponentialLinearRegression(sklearn.base.RegressorMixin):
    def __init__(self, *args, **kwargs):
        self.regr = Ridge(*args, **kwargs)

    def fit(self, X, Y):
        assert X is not None, "X can't be None"
        assert Y is not None, "Y can't be None"
        
        self.regr.fit(X, np.log(Y))
        return self

    def predict(self, X):
        assert X is not None, "X can't be None"
        return np.exp(self.regr.predict(X))
    
    def get_params(self, *args, **kwargs):
        return self.regr.get_params(*args, **kwargs)

    def set_params(self, **params):
        self.regr.set_params(**params)
        return self