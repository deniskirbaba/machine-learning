import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        if X is not None and y is not None:
            y = pd.DataFrame(y, columns=['average_bill'])
            data = pd.concat([X, y], axis=1)
            self.city_means_ = data.groupby(by='city')['average_bill'].mean()
            self.is_fitted_ = True
        return self

    def predict(self, X=None):
        if X is not None:
            return X['city'].map(self.city_means_).fillna(500)
        return None