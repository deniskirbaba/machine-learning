import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        if X is not None and y is not None:
            y = pd.DataFrame(y, columns=['average_bill'])
            design_matrix = pd.concat([X, y], axis=1)
            self.medians_ = design_matrix.groupby(['city', 'modified_rubrics'])['average_bill'].median().reset_index()
            self.is_fitted_ = True
            
        return self
    
    def predict(self, X=None):
        if X is not None:
            data = X[['modified_rubrics', 'city']]
            merged_data = data.merge(self.medians_, on=['modified_rubrics', 'city'], how='left')
            return merged_data['average_bill'].fillna(500)
        else:
            return None