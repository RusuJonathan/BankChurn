import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn import set_config

from src.preprocessing.preprocessing_utils import (
    add_financial_features,
    add_tenure_features,
    add_risk_features,
    add_interaction_features,
    SegmentStats,
)
from src.data.data_loader import load_yaml, preprocessing_path, ID_COLS

set_config(transform_output="pandas")

config = load_yaml(preprocessing_path)


class DropIdColumns(BaseEstimator, TransformerMixin):
    """Drops identifier columns (id, CustomerId, Surname)."""

    def __init__(self, id_cols: list = None):
        self.id_cols = id_cols or ID_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.id_cols if c in X.columns])


class FinancialFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_financial_features(X)


class TenureFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_tenure_features(X)


class RiskFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._balance_median = None

    def fit(self, X, y=None):
        self._balance_median = X["Balance"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["_balance_median_ref"] = self._balance_median
        result = add_risk_features(X)
        return result.drop(columns=["_balance_median_ref"], errors="ignore")


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_interaction_features(X)


class SegmentStatsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._segment_stats = SegmentStats()

    def fit(self, X, y=None):
        self._segment_stats.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._segment_stats.transform(X)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes Geography and Geography_x_Gender with OrdinalEncoder,
    Gender to binary (Male=1), then drops remaining object columns.
    """

    def __init__(self):
        self._geo_encoder  = OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1)
        self._cross_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                              unknown_value=-1)

    def fit(self, X, y=None):
        if "Geography" in X.columns:
            self._geo_encoder.fit(X[["Geography"]])
        if "Geography_x_Gender" in X.columns:
            self._cross_encoder.fit(X[["Geography_x_Gender"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Geography" in X.columns:
            X["Geography"] = self._geo_encoder.transform(X[["Geography"]])
        if "Gender" in X.columns:
            X["Gender"] = (X["Gender"] == "Male").astype(int)
        if "Geography_x_Gender" in X.columns:
            X["Geography_x_Gender"] = self._cross_encoder.transform(
                X[["Geography_x_Gender"]])
        obj_cols = X.select_dtypes(include="object").columns
        return X.drop(columns=obj_cols, errors="ignore")


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Median imputation for any remaining NaN values."""

    def __init__(self):
        self._imputer = SimpleImputer(strategy="median")

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include="number").columns
        self._num_cols = num_cols.tolist()
        self._imputer.fit(X[num_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._num_cols] = self._imputer.transform(X[self._num_cols])
        return X
