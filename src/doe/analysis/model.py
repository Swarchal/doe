"""Build a model (design) matrix from a :class:`~doe.design.Design`."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..design import Design
from ..factors import CategoricalFactor, ContinuousFactor, Factor, FactorSet

#: A factor's encoded model columns, as ``(term_name, column)`` pairs. A continuous
#: factor contributes one; a ``k``-level categorical factor contributes ``k - 1``.
_Encoding = list[tuple[str, np.ndarray]]


@dataclass
class ModelMatrix:
    """A model matrix: the intercept plus requested term columns, in coded units.

    Examples:
        >>> import numpy as np
        >>> matrix = ModelMatrix(
        ...     np.array([[1.0, -1.0], [1.0, 1.0]]),
        ...     ["Intercept", "temperature"],
        ... )
        >>> matrix.as_frame().to_dict("list")
        {'Intercept': [1.0, 1.0], 'temperature': [-1.0, 1.0]}
    """

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


def _categorical_coded_levels(factor: CategoricalFactor) -> np.ndarray:
    """Numeric coordinates used for categorical levels in candidate regions."""
    return np.linspace(-1.0, 1.0, len(factor.levels))


def _decode_categorical_coded(factor: CategoricalFactor, coded: np.ndarray) -> np.ndarray:
    """Map categorical candidate-region coordinates back to natural level labels."""
    coded = np.asarray(coded, dtype=float)
    levels = _categorical_coded_levels(factor)
    nearest = np.abs(coded[:, None] - levels[None, :]).argmin(axis=1)
    valid = np.isclose(coded, levels[nearest], atol=1e-9, rtol=0.0)
    if not np.all(valid):
        bad = np.unique(coded[~valid])
        raise ValueError(
            f"factor {factor.name!r} categorical candidate coordinate(s) "
            f"{bad.tolist()} do not match discrete coded levels {levels.tolist()}"
        )
    return np.asarray([factor.levels[int(i)] for i in nearest], dtype=object)


def coded_design_points(design: Design) -> np.ndarray:
    """Return design runs as numeric coded coordinates, including categorical factors.

    Continuous factors use their usual ``[-1, +1]`` coding. Categorical factors use the same
    evenly spaced candidate-region coordinates as :func:`expand_coded_points`, so a
    label-bearing :class:`Design` and a numeric candidate region can be stacked and expanded
    through one term layout.

    Examples:
        >>> import pandas as pd
        >>> from doe import CategoricalFactor, ContinuousFactor, Design, FactorSet
        >>> factors = FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ...     CategoricalFactor("catalyst", ("A", "B", "C")),
        ... ])
        >>> design = Design(
        ...     pd.DataFrame({"temperature": [40, 80], "catalyst": ["A", "C"]}),
        ...     factors,
        ... )
        >>> coded_design_points(design).tolist()
        [[-1.0, -1.0], [1.0, 1.0]]
    """
    coded = design.coded()
    cols: list[np.ndarray] = []
    for factor in design.factors:
        values = coded[factor.name].to_numpy()
        if isinstance(factor, CategoricalFactor):
            level_to_code = dict(zip(factor.levels, _categorical_coded_levels(factor), strict=True))
            try:
                cols.append(np.asarray([level_to_code[value] for value in values], dtype=float))
            except KeyError as exc:
                raise ValueError(
                    f"factor {factor.name!r} has unknown level {exc.args[0]!r}; "
                    f"expected {list(factor.levels)}"
                ) from exc
        else:
            cols.append(np.asarray(values, dtype=float))
    return np.column_stack(cols)


def _scheffe_matrix(points: np.ndarray, names: list[str], order: int) -> ModelMatrix:
    """Scheffé blending model matrix for mixture proportions.

    Because the proportions sum to 1, an intercept would be exactly collinear with the sum
    of the linear terms, and a squared term ``x_i^2 = x_i (1 - sum_{j!=i} x_j)`` is a linear
    combination of the linear and cross terms -- so the Scheffé form drops both:

    * ``order=1`` (linear blending): ``y-hat = sum_i b_i x_i``
    * ``order=2`` (quadratic blending): adds the ``i < j`` cross products ``b_ij x_i x_j``

    Term names are the component names and ``A:B`` products, consistent with the standard
    naming, so ANOVA/VIF/plots that key off ``term_names`` work unchanged.
    """
    cols: list[np.ndarray] = [points[:, j] for j in range(points.shape[1])]
    term_names = list(names)
    if order >= 2:
        for i, j in itertools.combinations(range(points.shape[1]), 2):
            cols.append(points[:, i] * points[:, j])
            term_names.append(f"{names[i]}:{names[j]}")
    return ModelMatrix(np.column_stack(cols), term_names)


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

    An all-mixture design takes the Scheffé path instead (see :func:`_scheffe_matrix`):
    no intercept, ``order`` selects linear vs quadratic blending, and ``interactions`` is
    ignored (mixture cross products are part of the quadratic blending model, not an
    independent choice).

    Examples:
        Build a main-effects-plus-interaction model for a mixed continuous/categorical
        design. The categorical factor is effect-coded before forming the interaction.

        >>> import pandas as pd
        >>> from doe import CategoricalFactor, ContinuousFactor, Design, FactorSet
        >>> factors = FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ...     CategoricalFactor("catalyst", ("A", "B")),
        ... ])
        >>> design = Design(
        ...     pd.DataFrame({
        ...         "temperature": [40, 80, 40, 80],
        ...         "catalyst": ["A", "A", "B", "B"],
        ...     }),
        ...     factors,
        ... )
        >>> build_model_matrix(design).term_names
        ['Intercept', 'temperature', 'catalyst[B]', 'temperature:catalyst[B]']

        A center point creates a continuous squared term when ``order=2``.

        >>> center = Design(pd.DataFrame({"temperature": [40, 60, 80]}), FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ... ]))
        >>> build_model_matrix(center, order=2, interactions=False).term_names
        ['Intercept', 'temperature', 'temperature^2']

        Mixture designs use the Scheffé no-intercept form directly on proportions.

        >>> from doe import MixtureFactor
        >>> blend = Design(pd.DataFrame({"A": [1.0, 0.5], "B": [0.0, 0.5]}), FactorSet([
        ...     MixtureFactor("A"),
        ...     MixtureFactor("B"),
        ... ]))
        >>> build_model_matrix(blend, order=2).term_names
        ['A', 'B', 'A:B']
    """
    coded = design.coded()
    if design.factors.is_mixture:
        points = coded.to_numpy(dtype=float)
        return _scheffe_matrix(points, design.factors.names, order)
    encodings: list[tuple[Factor, _Encoding]] = [
        (factor, _encode_factor(factor, coded[factor.name].to_numpy())) for factor in design.factors
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


def expand_coded_points(
    points: np.ndarray,
    factors: FactorSet,
    *,
    order: int = 1,
    interactions: bool = True,
) -> ModelMatrix:
    """Expand an ``(m, k)`` array of coded factor points into a model matrix.

    The array-based companion to :func:`build_model_matrix`: same intercept + main-effect +
    interaction (+ optional quadratic) expansion and identical ``term_names``, but operating on
    an arbitrary block of coded points rather than a :class:`~doe.design.Design`'s runs. This is
    the shared core the coordinate-exchange engine (I-optimality) and the G/I-efficiency region
    sampler use to evaluate the scaled prediction variance ``f(x)^T (X^T X)^-1 f(x)`` at
    candidate points ``x`` -- neither has a ``Design`` to hand, only coded coordinates.

    Categorical factors use discrete numeric coordinates in the candidate region:
    ``np.linspace(-1, 1, n_levels)`` maps to the factor's natural levels in order. Those labels
    are then passed through the same deviation/effect coding as :func:`build_model_matrix`, so
    mixed continuous/categorical designs and candidate regions share one term layout.
    ``points`` is ordered to match ``factors``; columns line up with
    ``build_model_matrix(design, order, interactions)``.

    Examples:
        >>> import numpy as np
        >>> from doe import CategoricalFactor, ContinuousFactor, FactorSet
        >>> factors = FactorSet([
        ...     ContinuousFactor("temperature", 40, 80),
        ...     CategoricalFactor("catalyst", ("A", "B")),
        ... ])
        >>> points = np.array([[-1.0, -1.0], [1.0, 1.0]])
        >>> expand_coded_points(points, factors).term_names
        ['Intercept', 'temperature', 'catalyst[B]', 'temperature:catalyst[B]']
        >>> expand_coded_points(points, factors).X.tolist()
        [[1.0, -1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2-D array with shape (n_points, n_factors)")
    if points.shape[1] != len(factors):
        raise ValueError(
            f"points has {points.shape[1]} columns but factors has {len(factors)} entries"
        )

    # mixture candidate points are proportions; expand them through the same Scheffé
    # path as build_model_matrix so the optimal-design engine scores the right model.
    if factors.is_mixture:
        return _scheffe_matrix(points, factors.names, order)

    encodings: list[tuple[Factor, _Encoding]] = []
    for j, factor in enumerate(factors):
        if isinstance(factor, CategoricalFactor):
            labels = _decode_categorical_coded(factor, points[:, j])
            encodings.append((factor, _effect_code(factor, labels)))
        else:
            encodings.append((factor, [(factor.name, points[:, j])]))

    cols: list[np.ndarray] = [np.ones(points.shape[0])]
    term_names: list[str] = ["Intercept"]

    for _factor, encoding in encodings:
        for name, col in encoding:
            cols.append(col)
            term_names.append(name)

    if interactions:
        for (_fa, enc_a), (_fb, enc_b) in itertools.combinations(encodings, 2):
            for (name_a, col_a), (name_b, col_b) in itertools.product(enc_a, enc_b):
                cols.append(col_a * col_b)
                term_names.append(f"{name_a}:{name_b}")

    if order >= 2:
        for factor, encoding in encodings:
            if not isinstance(factor, ContinuousFactor):
                continue
            (name, col) = encoding[0]
            if np.any(np.abs(np.abs(col) - 1.0) > 1e-9):
                cols.append(col**2)
                term_names.append(f"{name}^2")

    return ModelMatrix(np.column_stack(cols), term_names)
