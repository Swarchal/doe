"""DoE -- a Python library for design-of-experiment analysis.

Phase 1 public API: factor definitions, the Design container, factorial generators,
and OLS analysis. See ``docs/PLAN.md`` for the full roadmap.
"""

from __future__ import annotations

from .analysis.anova import (
    LackOfFit,
    adjusted_r2,
    anova_table,
    lack_of_fit,
    predicted_r2,
    press,
)
from .analysis.fit import FitResult, fit_ols
from .analysis.optimize import (
    DesirabilityResult,
    Optimum,
    ResponseGoal,
    StationaryPoint,
    desirability,
    optimum,
    stationary_point,
)
from .design import Design
from .factors import CategoricalFactor, ContinuousFactor, Factor, FactorSet
from .generators.factorial import (
    fractional_factorial,
    full_factorial,
    plackett_burman,
)
from .generators.rsm import box_behnken, central_composite
from .interactive import to_html

__all__ = [
    "CategoricalFactor",
    "ContinuousFactor",
    "Design",
    "DesirabilityResult",
    "Factor",
    "FactorSet",
    "FitResult",
    "LackOfFit",
    "Optimum",
    "ResponseGoal",
    "StationaryPoint",
    "adjusted_r2",
    "anova_table",
    "box_behnken",
    "central_composite",
    "desirability",
    "fit_ols",
    "fractional_factorial",
    "full_factorial",
    "lack_of_fit",
    "optimum",
    "plackett_burman",
    "predicted_r2",
    "press",
    "stationary_point",
    "to_html",
]

__version__ = "0.1.0"
