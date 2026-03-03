import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.models.models_builders import (
    build_pipeline,
    build_calibrated_pipeline,
    build_stacking_model,
    set_base_model_params,
)
from src.models.hpo_tuner import run_hyperparameter_optimization, save_best_hyperparameters
from src.data.data_loader import load_yaml, load_data, train_path, model_config_path


class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, config: Dict, n_trials: int = 300):
        self.config   = config
        self.n_trials = n_trials
        self._init_models()

    def _init_models(self):
        self.lgbm = (
            "lgbm",
            build_calibrated_pipeline(LGBMClassifier,
                                      {"verbosity": -1, "n_jobs": -1, "eval_metric": "auc"},
                                      method="isotonic"),
        )
        self.xgboost = (
            "xgboost",
            build_calibrated_pipeline(XGBClassifier,{"verbosity": 0, "n_jobs": -1,"eval_metric": "auc"},
                                      method="isotonic"),
        )
        self.catboost = (
            "catboost",
            build_pipeline(CatBoostClassifier, {"verbose": 0, "eval_metric": "AUC"}),
        )
        self.randomforest = (
            "random_forest",
            build_calibrated_pipeline(RandomForestClassifier,{"n_jobs": -1},
                                      method="isotonic"),
        )
        self.logistic = (
            "logistic",
            build_pipeline(LogisticRegression,
                           {"max_iter": 5000, "solver": "saga", "l1_ratio": 0.5}),
        )

        self.estimators = [
            # self.lgbm,
            self.xgboost,
            self.catboost,
            # self.randomforest,
            # self.logistic,
        ]

        self.stacking_model = build_stacking_model(
            estimators=self.estimators,
            final_estimator=LogisticRegression(
                max_iter=5000, solver="saga", l1_ratio=0.5, C=1.0,
            ),
        )

    def optimize_hyperparameters(
            self,
            X: pd.DataFrame,
            y: pd.Series) -> None:
        for model_name, pipeline in self.estimators:
            if model_name not in self.config:
                print(f"Skipping {model_name} (no config found)")
                continue
            print(f"\n🔍 Optimising {model_name} ({self.n_trials} trials)...")
            study = run_hyperparameter_optimization(
                model=pipeline,
                X=X,
                y=y,
                model_config=self.config[model_name],
                n_trials=self.n_trials,
            )
            set_base_model_params(pipeline, study.best_params)
            save_best_hyperparameters(study=study, model_name=model_name)
            print(f"  Best AUC: {study.best_value:.4f}")

    def load_hyperparameters(self, hyperparameter_config: Dict) -> "Model":
        for model_name, pipeline in self.estimators:
            if model_name in hyperparameter_config:
                set_base_model_params(pipeline, hyperparameter_config[model_name])
        return self

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series) -> "Model":
        self.stacking_model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.stacking_model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.stacking_model.predict(X)

    def cross_validate(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            cv: int = 5) -> Dict:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            estimator=self,
            X=X,
            y=y,
            scoring="roc_auc",
            cv=skf,
        )
        return {"mean_auc": scores.mean(), "std_auc": scores.std(), "scores": scores}


if __name__ == "__main__":
    config  = load_yaml(model_config_path)
    df      = load_data(train_path)
    x_train = df.drop(columns=["Exited"])
    y_train = df["Exited"]

    model = Model(config=config, n_trials=50)
    model.optimize_hyperparameters(x_train, y_train)
    results = model.cross_validate(x_train, y_train)
    print(f"\nCV AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
