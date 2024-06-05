import numpy as np
import pandas as pd
from typing import Optional, List

import sklearn.base
from sklearn.preprocessing import StandardScaler


class BaseDataPreprocessor(sklearn.base.TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        if self.needed_columns:
            data = data.loc[:, self.needed_columns]
        data = data.to_numpy()
        self.scaler.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        if self.needed_columns:
            data = data.loc[:, self.needed_columns]
        data = data.to_numpy()
        return self.scaler.transform(data)