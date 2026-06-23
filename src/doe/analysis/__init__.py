"""Analysis routines. Phase 1: model matrix building and OLS fitting."""

from __future__ import annotations

from .anova import (
    LackOfFit,
    adjusted_r2,
    anova_table,
    lack_of_fit,
    predicted_r2,
    press,
)
from .fit import FitResult, fit_ols
from .model import ModelMatrix, build_model_matrix

__all__ = [
    "FitResult",
    "LackOfFit",
    "ModelMatrix",
    "adjusted_r2",
    "anova_table",
    "build_model_matrix",
    "fit_ols",
    "lack_of_fit",
    "predicted_r2",
    "press",
]
