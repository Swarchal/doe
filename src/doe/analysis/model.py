"""Build a model (design) matrix from a :class:`~doe.design.Design`."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..design import Design
from ..factors import CategoricalFactor


@dataclass
class ModelMatrix:
    """A model matrix: the intercept plus requested term columns, in coded units."""

    X: np.ndarray
    term_names: list[str]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, columns=self.term_names)


def build_model_matrix(design: Design, order: int = 1, interactions: bool = True) -> ModelMatrix:
    """Build a coded model matrix.

    Args:
        design: the design to expand.
        order: highest single-factor power (1 = linear; 2 adds squared terms).
        interactions: include 2-factor interaction (product) terms.

    Categorical contrast expansion is deferred to Phase 2; categorical factors are
    rejected explicitly until then.
    """
    categorical = [
        factor.name for factor in design.factors if isinstance(factor, CategoricalFactor)
    ]
    if categorical:
        raise NotImplementedError(
            "categorical factors require contrast expansion, which is deferred to Phase 2; "
            f"unsupported factors: {categorical}"
        )

    coded = design.coded()
    names = list(coded.columns)
    cols: list[np.ndarray] = [np.ones(len(coded))]
    term_names: list[str] = ["Intercept"]

    for name in names:
        cols.append(coded[name].to_numpy(dtype=float))
        term_names.append(name)

    if interactions:
        for a, b in itertools.combinations(names, 2):
            cols.append(coded[a].to_numpy(float) * coded[b].to_numpy(float))
            term_names.append(f"{a}:{b}")

    if order >= 2:
        for name in names:
            col = coded[name].to_numpy(float)
            # a pure +/-1 factor has x^2 == 1 (collinear with the intercept); only emit a
            # squared term once the factor actually takes a value off {-1, +1}.
            if np.any(np.abs(np.abs(col) - 1.0) > 1e-9):
                cols.append(col**2)
                term_names.append(f"{name}^2")

    return ModelMatrix(np.column_stack(cols), term_names)
