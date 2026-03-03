import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.data.data_loader import load_yaml, preprocessing_path

config = load_yaml(preprocessing_path)


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates financially-motivated derived features.

    - BalanceToSalary : wealth concentration ratio — high values signal
                            either savings-focused customers or salary under-reporters
    - CreditScorePerAge : creditworthiness normalised by life stage;
                            a 25yr old with 750 is more remarkable than a 60yr old
    - BalancePerProduct : average balance per product used — high = premium customer
    - SalaryPerProduct : estimated revenue potential per product
    - BalanceIsZero: binary flag — zero balance customers often churn silently
    """
    df = df.copy()

    safe_salary = df["EstimatedSalary"].replace(0, np.nan)
    safe_age = df["Age"].replace(0, np.nan)
    safe_prods = df["NumOfProducts"].replace(0, np.nan)

    df["BalanceToSalary"] = df["Balance"]       / safe_salary
    df["CreditScorePerAge"] = df["CreditScore"]   / safe_age
    df["BalancePerProduct"] = df["Balance"]        / safe_prods
    df["SalaryPerProduct"] = df["EstimatedSalary"] / safe_prods
    df["BalanceIsZero"] = (df["Balance"] == 0).astype(int)

    ratio_cols = ["BalanceToSalary", "CreditScorePerAge",
                  "BalancePerProduct", "SalaryPerProduct"]
    df[ratio_cols] = df[ratio_cols].fillna(0)

    return df

def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretises Tenure into behavioural groups.
    Customer lifecycle stages matter for churn modelling:
      - New (0-2y)   : still evaluating, high volatility
      - Mid (3-7y)   : established relationship
      - Loyal (8-10y): either very satisfied or inertia-locked
    """
    df = df.copy()

    df["IsNew"] = (df["Tenure"] <= 1).astype(int)
    df["IsLoyal"] = (df["Tenure"] >= 8).astype(int)
    df["TenureGroup"] = pd.cut(
        df["Tenure"],
        bins=[-1, 2, 7, 10],
        labels=[0, 1, 2],
    ).astype(int)

    return df

def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates composite risk and engagement indicators.

    - AgeRiskScore         : age bands where churn is empirically higher
                             (35-55 is the peak churn window in banking)
    - InactiveRichCustomer : dormant high-balance customer — highest churn risk
    - LowCreditHighBalance : potential credit stress
    - EngagementScore      : composite loyalty proxy
    """
    df = df.copy()

    # Age risk: 35-55 is the high-mobility career phase
    df["AgeRiskScore"] = np.where(
        (df["Age"] >= 35) & (df["Age"] <= 55), 2,
        np.where(df["Age"] > 55, 1, 0)
    )

    balance_median = df["Balance"].median()
    df["InactiveRichCustomer"] = ((df["IsActiveMember"] == 0) & (df["Balance"] > balance_median)).astype(int)

    df["LowCreditHighBalance"] = (
        (df["CreditScore"] < 500) & (df["Balance"] > balance_median)
    ).astype(int)

    df["EngagementScore"] = (
        df["NumOfProducts"] +
        df["IsActiveMember"] +
        df["HasCrCard"]
    )

    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    adds interaction features
    """

    df = df.copy()

    df["Age_x_NumProducts"] = df["Age"]* df["NumOfProducts"]
    df["CreditScore_x_Balance"] = df["CreditScore"] * np.log1p(df["Balance"])
    df["Geography_x_Gender"] = df["Geography"] + "_" + df["Gender"]

    return df


def log_transform_features(df: pd.DataFrame,
                             features: List[str]) -> pd.DataFrame:
    """applies log1p to right-skewed columns"""
    df = df.copy()
    for col in features:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df

class SegmentStats:
    """
    Computes segment-level statistics that simulate lag / rolling features
    for tabular data where no explicit time index per customer exists.

    For each grouping key, we compute:
      - median and std of the target feature within the segment
    These are fitted exclusively on training data to prevent leakage.
    """

    def __init__(self):
        self._stats: Dict[str, pd.DataFrame] = {}
        self._global_fallbacks: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "SegmentStats":
        df = df.copy()

        for feat in ["CreditScore", "Balance"]:
            agg = df.groupby("Geography")[feat].agg(["median", "std"]).rename(
                columns={"median": f"{feat}_geo_median",
                         "std":    f"{feat}_geo_std"}
            )
            self._stats[f"{feat}_by_geo"] = agg
            self._global_fallbacks[f"{feat}_geo_median"] = df[feat].median()
            self._global_fallbacks[f"{feat}_geo_std"]    = df[feat].std()

        agg = df.groupby("NumOfProducts")["Age"].median().rename("Age_product_median")
        self._stats["Age_by_products"] = agg.to_frame()
        self._global_fallbacks["Age_product_median"] = df["Age"].median()

        df["_geo_prod"] = df["Geography"] + "_" + df["NumOfProducts"].astype(str)
        agg = df.groupby("_geo_prod")["EstimatedSalary"].median().rename("Salary_geo_product_median")
        self._stats["Salary_by_geo_prod"] = agg.to_frame()
        self._global_fallbacks["Salary_geo_product_median"] = df["EstimatedSalary"].median()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for feat in ["CreditScore", "Balance"]:
            key = f"{feat}_by_geo"
            stats_df = self._stats[key]
            df = df.merge(stats_df, how="left", left_on="Geography", right_index=True)
            df[f"{feat}_geo_median"] = df[f"{feat}_geo_median"].fillna(
                self._global_fallbacks[f"{feat}_geo_median"])
            df[f"{feat}_geo_std"] = df[f"{feat}_geo_std"].fillna(
                self._global_fallbacks[f"{feat}_geo_std"])

        df = df.merge(
            self._stats["Age_by_products"],
            how="left", left_on="NumOfProducts", right_index=True)
        df["Age_product_median"] = df["Age_product_median"].fillna(
            self._global_fallbacks["Age_product_median"])

        df["_geo_prod"] = df["Geography"] + "_" + df["NumOfProducts"].astype(str)
        df = df.merge(
            self._stats["Salary_by_geo_prod"],
            how="left", left_on="_geo_prod", right_index=True)
        df["Salary_geo_product_median"] = df["Salary_geo_product_median"].fillna(
            self._global_fallbacks["Salary_geo_product_median"])
        df = df.drop(columns=["_geo_prod"], errors="ignore")

        return df
