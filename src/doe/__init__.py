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
from .analysis.diagnostics import (
    Efficiency,
    condition_number,
    correlation_matrix,
    discrepancy,
    efficiency,
    information_matrix,
    leverage,
    log_det_information,
    maximin_distance,
    vif,
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
from .factors import (
    CategoricalFactor,
    ContinuousFactor,
    Factor,
    FactorSet,
    factor_from_dict,
)
from .generators.factorial import (
    fractional_factorial,
    full_factorial,
    plackett_burman,
)
from .generators.optimal import (
    OptimalDesign,
    augment,
    candidate_grid,
    coordinate_exchange,
    d_optimal,
    i_optimal,
)
from .generators.rsm import box_behnken, central_composite
from .generators.spacefilling import halton, latin_hypercube, sobol
from .interactive import to_html
from .serialization import ValidationError, validate_design_dict

__all__ = [
    "CategoricalFactor",
    "ContinuousFactor",
    "Design",
    "DesirabilityResult",
    "Efficiency",
    "Factor",
    "FactorSet",
    "FitResult",
    "LackOfFit",
    "Optimum",
    "OptimalDesign",
    "ResponseGoal",
    "StationaryPoint",
    "ValidationError",
    "adjusted_r2",
    "anova_table",
    "augment",
    "box_behnken",
    "candidate_grid",
    "central_composite",
    "condition_number",
    "coordinate_exchange",
    "correlation_matrix",
    "d_optimal",
    "desirability",
    "discrepancy",
    "efficiency",
    "factor_from_dict",
    "fit_ols",
    "fractional_factorial",
    "full_factorial",
    "halton",
    "i_optimal",
    "information_matrix",
    "lack_of_fit",
    "latin_hypercube",
    "leverage",
    "log_det_information",
    "maximin_distance",
    "optimum",
    "plackett_burman",
    "predicted_r2",
    "press",
    "sobol",
    "stationary_point",
    "to_html",
    "validate_design_dict",
    "vif",
]

__version__ = "0.1.0"
