"""Shared, untuned Logistic Regression definition and probability metrics."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONTINUOUS_COLUMNS = (
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "annual_precip_mean", "evt_dominant_fraction",
)
ASPECT_COLUMNS = ("aspect_sin_mean", "aspect_cos_mean", "aspect_strength")
CATEGORICAL_COLUMNS = ("evt_dominant_class",)
PREDICTORS = (*CONTINUOUS_COLUMNS, *ASPECT_COLUMNS, *CATEGORICAL_COLUMNS)


def build_logistic_pipeline() -> Pipeline:
    """Create a fresh baseline whose preprocessing learns only from fitted rows.

    Undefined flat-terrain aspect is replaced with zero before scaling. EVT
    codes are nominal categories; unseen validation categories encode as zeros.
    All other numeric predictors must already be nonmissing. The model uses
    L2 regularization, C=1, no class weighting, and no hyperparameter tuning.

    Returns:
        Unfitted preprocessing and Logistic Regression pipeline.
    """
    aspect = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("continuous", StandardScaler(), list(CONTINUOUS_COLUMNS)),
        ("aspect", aspect, list(ASPECT_COLUMNS)),
        ("vegetation", OneHotEncoder(handle_unknown="ignore", sparse_output=True), list(CATEGORICAL_COLUMNS)),
    ], remainder="drop", sparse_threshold=1.0)
    # In sklearn 1.9, l1_ratio=0 specifies L2 without the deprecated penalty argument.
    model = LogisticRegression(
        solver="lbfgs", C=1.0, l1_ratio=0.0, max_iter=2000, tol=1e-4,
        class_weight=None, fit_intercept=True, random_state=42, warm_start=False,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def calculate_probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Measure ranking performance without choosing a classification threshold.

    Args:
        target: One-dimensional binary labels containing both classes.
        probability: Matching class-1 probabilities.

    Returns:
        ROC-AUC and pr_auc, defined as Average Precision (not trapezoidal area).

    Raises:
        ValueError: If labels or probabilities are invalid or misaligned.
    """
    target = np.asarray(target)
    probability = np.asarray(probability)
    if target.ndim != 1 or probability.shape != target.shape or set(target) != {0, 1}:
        raise ValueError("Metrics require matching one-dimensional probabilities and both binary classes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return {"roc_auc": float(roc_auc_score(target, probability)),
            "pr_auc": float(average_precision_score(target, probability))}
