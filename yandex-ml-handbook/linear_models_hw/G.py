import numpy as np
import pandas as pd

from typing import Optional, List

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class BaseDataPreprocessor(TransformerMixin):
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


class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, categorical_columns: Optional[List[str]]=None, **kwargs):
        super().__init__(**kwargs)
        self.categorical_columns = categorical_columns
        self.onehotenc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        
    def fit(self, data: pd.DataFrame, *args):
        super().fit(data, *args)
        
        if self.categorical_columns:
            data = data.loc[:, self.categorical_columns]
        self.onehotenc.fit(data)
        
        return self

    def transform(self, data: pd.DataFrame):
        cont_data = super().transform(data)
        
        if self.categorical_columns:
            data = data.loc[:, self.categorical_columns]
        cat_data = self.onehotenc.transform(data)
                    
        return np.concatenate((cont_data, cat_data), axis=1)
    
    
continuous_columns = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add', 'Mas_Vnr_Area', 'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Low_Qual_Fin_SF', 'Gr_Liv_Area', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'TotRms_AbvGrd', 'Fireplaces', 'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch', 'Screen_Porch', 'Pool_Area', 'Misc_Val', 'Mo_Sold', 'Year_Sold', 'Longitude', 'Latitude']
interesting_columns = ["Overall_Qual", "Garage_Qual", "Sale_Condition", "MS_Zoning"]

def make_ultimate_pipeline(continuous_columns, interesting_columns):
    return Pipeline([('preproc', OneHotPreprocessor(needed_columns=continuous_columns, categorical_columns=interesting_columns)), ('ridge', Ridge())])
