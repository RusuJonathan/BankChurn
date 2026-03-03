from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config

from src.preprocessing.transformers import (
    DropIdColumns,
    FinancialFeatureTransformer,
    TenureFeatureTransformer,
    RiskFeatureTransformer,
    InteractionFeatureTransformer,
    SegmentStatsTransformer,
    CategoricalEncoder,
    NumericalImputer,
)

set_config(transform_output="pandas")


def build_data_preparation_pipeline() -> Pipeline:
    """
    Full preprocessing pipeline — applied identically to train and test.

    Step-by-step:
      1.  DropIdColumns              — removes id, CustomerId, Surname
      2.  FinancialFeatureTransformer— BalanceToSalary, CreditScorePerAge, …
      3.  TenureFeatureTransformer   — IsNew, IsLoyal, TenureGroup
      4.  RiskFeatureTransformer     — AgeRiskScore, InactiveRichCustomer, …
                                       (balance median fitted on train only)
      5.  InteractionFeatureTransformer — Age×NumProducts, CreditScore×Balance,
                                          Geography×Gender cross-feature
      6.  SegmentStatsTransformer    — geo/product segment medians & stds
                                       (fitted on train only — no leakage)
      7.  CategoricalEncoder         — Geography ordinal, Gender binary, cross-enc
      8.  NumericalImputer           — median imputation for any residual NaN
      9. StandardScaler             — zero-mean, unit-variance
    """
    return Pipeline(steps=[
        ("drop_ids",          DropIdColumns()),
        ("financial",         FinancialFeatureTransformer()),
        ("tenure",            TenureFeatureTransformer()),
        ("risk",              RiskFeatureTransformer()),
        ("interactions",      InteractionFeatureTransformer()),
        ("segment_stats",     SegmentStatsTransformer()),
        ("cat_encoder",       CategoricalEncoder()),
        ("imputer",           NumericalImputer()),
        ("scaler",            StandardScaler()),
    ])
