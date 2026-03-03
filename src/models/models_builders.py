from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Tuple, Any

from src.preprocessing.pipeline import build_data_preparation_pipeline


def build_pipeline(
        model_class: Any,
        param: Dict[str, Any] = {}) -> Pipeline:

    return Pipeline(steps=[
        ("preprocessing", build_data_preparation_pipeline()),
        ("model", model_class(**param)),
    ])


def build_calibrated_pipeline(
        model_class: Any,
        param: Dict[str, Any] = {},
        method: str = "isotonic") -> Pipeline:

    base_model = model_class(**param)
    calibrated = CalibratedClassifierCV(base_model, method=method, cv=5)
    return Pipeline(steps=[
        ("preprocessing", build_data_preparation_pipeline()),
        ("model", calibrated),
    ])


def build_stacking_model(
        estimators: List[Tuple[str, Any]],
        final_estimator: Any,
        cv: int = 5,
        stack_method: str = "predict_proba") -> StackingClassifier:

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        stack_method=stack_method,
        n_jobs=-1,
        passthrough=False,
    )


def set_base_model_params(
        pipeline: Pipeline,
        param: Dict[str, Any]) -> None:
    """Sets hyperparameters on the model step of a pipeline."""
    model_step = pipeline.named_steps["model"]
    # Handle CalibratedClassifierCV wrapper
    if hasattr(model_step, "estimator"):
        model_step.estimator.set_params(**param)
    else:
        model_step.set_params(**param)


def set_stacking_model_params(
        stacking: StackingClassifier,
        param: Dict[str, Any]) -> None:
    stacking.final_estimator.set_params(**param)