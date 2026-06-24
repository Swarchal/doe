"""Build a model (design) matrix from a :class:`~doe.design.Design`."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..design import Design
from ..factors import CategoricalFactor, ContinuousFactor, Factor

#: A factor's encoded model columns, as ``(term_name, column)`` pairs. A continuous
#: factor contributes one; a ``k``-level categorical factor contributes ``k - 1``.
_Encoding = list[tuple[str, np.ndarray]]


@dataclass
class ModelMatrix:
    """A model matrix: the intercept plus requested term columns, in coded units."""

    X: np.ndarray
    term_names: list[str]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, columns=self.term_names)


def _effect_code(factor: CategoricalFactor, values: np.ndarray) -> _Encoding:
    """Deviation (effect) coding for a categorical factor.

    A ``k``-level factor becomes ``k - 1`` contrast columns. The *first* level is the
    reference (coded ``-1`` across every column); each remaining level gets a column that
    is ``+1`` for that level, ``-1`` for the reference, and ``0`` otherwise. For a 2-level
    factor this collapses to a single ``+/-1`` column matching the generator's corner
    coding (``levels[0]`` -> ``-1``, ``levels[1]`` -> ``+1``), so categorical and
    continuous main effects share the ``effect = 2 * coefficient`` interpretation.
    """
    levels = list(factor.levels)
    vals = np.asarray(values, dtype=object)
    unknown = set(vals.tolist()) - set(levels)
    if unknown:
        raise ValueError(
            f"factor {factor.name!r} has unknown level(s) "
            f"{sorted(map(str, unknown))}; expected {levels}"
        )

    reference = levels[0]
    is_reference = vals == reference
    encoding: _Encoding = []
    for level in levels[1:]:
        col = np.zeros(len(vals))
        col[vals == level] = 1.0
        col[is_reference] = -1.0
        encoding.append((f"{factor.name}[{level}]", col))
    return encoding


def _encode_factor(factor: Factor, values: np.ndarray) -> _Encoding:
    """Encode one factor into its model column(s)."""
    if isinstance(factor, CategoricalFactor):
        return _effect_code(factor, values)
    return [(factor.name, np.asarray(values, dtype=float))]


def build_model_matrix(design: Design, order: int = 1, interactions: bool = True) -> ModelMatrix:
    """Build a coded model matrix.

    Args:
        design: the design to expand.
        order: highest single-factor power (1 = linear; 2 adds squared terms).
        interactions: include 2-factor interaction (product) terms.

    Continuous factors map to a single ``[-1, +1]`` column. Categorical factors are
    expanded by deviation (effect) coding into ``k - 1`` contrast columns named
    ``factor[level]`` (see :func:`_effect_code`). Interactions are formed as the products
    of the participating factors' encoded columns, so a continuous-by-categorical or
    categorical-by-categorical interaction contributes one column per combination of
    contrasts. Squared terms are emitted only for continuous factors that actually take a
    value off ``{-1, +1}`` (a pure +/-1 column squares to the intercept).
    """
    coded = design.coded()
    encodings: list[tuple[Factor, _Encoding]] = [
        (factor, _encode_factor(factor, coded[factor.name].to_numpy()))
        for factor in design.factors
    ]

    cols: list[np.ndarray] = [np.ones(len(coded))]
    term_names: list[str] = ["Intercept"]

    for _factor, encoding in encodings:
        for name, col in encoding:
            cols.append(col)
            term_names.append(name)

    if interactions:
        # An interaction term captures synergy/antagonism: the effect of one factor depending on
        # the level of another. In coded units this is exactly the elementwise product of the two
        # factors' columns -- positive where both agree in sign, negative where they oppose.
        for (_fa, enc_a), (_fb, enc_b) in itertools.combinations(encodings, 2):
            for (name_a, col_a), (name_b, col_b) in itertools.product(enc_a, enc_b):
                cols.append(col_a * col_b)
                term_names.append(f"{name_a}:{name_b}")

    if order >= 2:
        # Squared terms model curvature -- a response that peaks or bottoms out in the interior
        # rather than rising monotonically. They turn the linear model into a second-order
        # (response-surface) model whose optimum can be located (see analysis.optimize).
        for factor, encoding in encodings:
            # squares apply only to continuous factors; a categorical contrast column
            # squared is not a meaningful curvature term.
            if not isinstance(factor, ContinuousFactor):
                continue
            (name, col) = encoding[0]
            # a pure +/-1 factor has x^2 == 1 (collinear with the intercept); only emit a
            # squared term once the factor actually takes a value off {-1, +1}.
            if np.any(np.abs(np.abs(col) - 1.0) > 1e-9):
                cols.append(col**2)
                term_names.append(f"{name}^2")

    return ModelMatrix(np.column_stack(cols), term_names)
