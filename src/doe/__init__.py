"""DoE -- a Python library for design-of-experiment analysis.

Public API: factor definitions, the Design container, factorial generators, and
OLS analysis. See ``docs/PLAN.md`` for the full roadmap.
"""

from __future__ import annotations

from .analysis.anova import (
    LackOfFit,
    adjusted_r2,
    anova_records,
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
from .analysis.fit import (
    MODEL_SPECS,
    FitResult,
    RankDeficientModelError,
    SaturatedFitWarning,
    fit_gls,
    fit_ols,
)
from .analysis.optimize import (
    CategoricalOptimum,
    DesirabilityResult,
    Optimum,
    ResponseGoal,
    StationaryPoint,
    categorical_optimum,
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
    MixtureFactor,
    factor_from_dict,
)
from .generators.blocking import (
    blocked_factorial,
    latin_square,
    randomized_complete_block,
)
from .generators.factorial import (
    fractional_factorial,
    full_factorial,
    plackett_burman,
)
from .generators.mixture import (
    extreme_vertices,
    mixture_candidates,
    simplex_centroid,
    simplex_lattice,
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
from .generators.screening import definitive_screening
from .generators.spacefilling import halton, latin_hypercube, sobol
from .generators.splitplot import split_plot
from .interactive import to_html
from .serialization import ValidationError, json_safe, validate_design_dict

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
    "MixtureFactor",
    "CategoricalOptimum",
    "Optimum",
    "OptimalDesign",
    "ResponseGoal",
    "MODEL_SPECS",
    "RankDeficientModelError",
    "SaturatedFitWarning",
    "StationaryPoint",
    "ValidationError",
    "adjusted_r2",
    "anova_records",
    "anova_table",
    "augment",
    "blocked_factorial",
    "box_behnken",
    "candidate_grid",
    "central_composite",
    "condition_number",
    "coordinate_exchange",
    "correlation_matrix",
    "d_optimal",
    "definitive_screening",
    "desirability",
    "discrepancy",
    "efficiency",
    "extreme_vertices",
    "factor_from_dict",
    "fit_gls",
    "fit_ols",
    "fractional_factorial",
    "full_factorial",
    "halton",
    "i_optimal",
    "information_matrix",
    "json_safe",
    "lack_of_fit",
    "latin_hypercube",
    "latin_square",
    "leverage",
    "log_det_information",
    "maximin_distance",
    "mixture_candidates",
    "categorical_optimum",
    "optimum",
    "plackett_burman",
    "predicted_r2",
    "press",
    "randomized_complete_block",
    "simplex_centroid",
    "simplex_lattice",
    "sobol",
    "split_plot",
    "stationary_point",
    "to_html",
    "validate_design_dict",
    "vif",
]

__version__ = "0.1.0"
