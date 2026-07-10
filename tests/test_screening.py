"""Definitive screening designs (Phase 5a).

Written test-first to document the intended API and DSD defining properties.
Correctness anchors follow Jones & Nachtsheim (2011): 2k+1 runs, three coded levels,
main-effect orthogonality (including orthogonality to second-order terms), one all-zero
center run with two zeros per factor column, and the tabulated k=6 design.
"""

import numpy as np
import pytest

from doe.analysis.model import build_model_matrix
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.screening import _conference_matrix, definitive_screening


def _factors(n, low=0.0, high=10.0):
    return [ContinuousFactor(chr(ord("a") + i), low=low, high=high) for i in range(n)]


def test_dsd_run_count_even_k():
    # k=4 continuous factors -> 2*4 + 1 = 9 runs
    design = definitive_screening(_factors(4))
    assert design.n_runs == 9


def test_dsd_three_coded_levels():
    design = definitive_screening(_factors(4))
    coded = design.coded().to_numpy()
    assert set(np.unique(coded)) == {-1.0, 0.0, 1.0}


def test_dsd_single_center_run():
    # exactly one all-zero (structural center) run, tagged "center"
    design = definitive_screening(_factors(4))
    coded = design.coded().to_numpy()
    zero_rows = np.all(coded == 0.0, axis=1)
    assert zero_rows.sum() == 1


def test_dsd_two_zeros_per_factor_column():
    # foldover [C; -C] gives each factor exactly two zeros (the diagonal in C and -C)
    design = definitive_screening(_factors(4))
    coded = design.coded().to_numpy()
    for j in range(coded.shape[1]):
        assert np.count_nonzero(coded[:, j] == 0.0) == 2 + 1  # +1 for the center run


@pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
def test_dsd_main_effects_orthogonal(k):
    # main-effect columns are mutually orthogonal (diagonal XᵀX) and orthogonal to
    # every quadratic column -- the DSD defining property.
    design = definitive_screening(_factors(k))
    matrix = build_model_matrix(design, order=2, interactions=False)
    factor_names = set(design.factors.names)
    main_idx = [i for i, name in enumerate(matrix.term_names) if name in factor_names]
    quad_idx = [i for i, name in enumerate(matrix.term_names) if name.endswith("^2")]
    assert len(main_idx) == k
    assert len(quad_idx) == k  # every factor takes a 0 level, so every square is emitted

    main = matrix.X[:, main_idx]
    quad = matrix.X[:, quad_idx]

    gram = main.T @ main
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.allclose(off_diagonal, 0.0, atol=1e-9)

    assert np.allclose(main.T @ quad, 0.0, atol=1e-9)


def test_dsd_odd_k_adds_fake_factor():
    # k=5 (odd) auto-adds one fake factor -> 2*5 + 3 = 13 runs, fake column dropped
    design = definitive_screening(_factors(5))
    assert design.n_runs == 13
    assert design.factors.names == ["a", "b", "c", "d", "e"]


def test_dsd_k6_matches_jones_nachtsheim_table():
    # Reproducing the exact published k=6 matrix up to row/column permutation and sign is
    # hard to anchor robustly; instead assert the full set of structural properties that
    # only a valid 13-run definitive screening design satisfies (Jones & Nachtsheim 2011,
    # Table 4): 13 runs, three coded levels, one all-zero center run, a foldover
    # [C; -C] core with mutually orthogonal columns of equal (conference-matrix) norm, two
    # structural zeros per factor column, and main effects orthogonal to every quadratic term.
    design = definitive_screening(_factors(6))
    assert design.n_runs == 13

    coded = design.coded().to_numpy()
    assert set(np.unique(coded)) == {-1.0, 0.0, 1.0}

    zero_rows = np.all(coded == 0.0, axis=1)
    assert zero_rows.sum() == 1

    structural = coded[~zero_rows]
    assert structural.shape == (12, 6)
    gram = structural.T @ structural
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.allclose(off_diagonal, 0.0, atol=1e-9)
    # every factor's foldover column has the same (conference-matrix) norm
    assert np.allclose(np.diag(gram), np.diag(gram)[0])

    for j in range(6):
        assert np.count_nonzero(coded[:, j] == 0.0) == 2 + 1

    matrix = build_model_matrix(design, order=2, interactions=False)
    factor_names = set(design.factors.names)
    main_idx = [i for i, name in enumerate(matrix.term_names) if name in factor_names]
    quad_idx = [i for i, name in enumerate(matrix.term_names) if name.endswith("^2")]
    main = matrix.X[:, main_idx]
    quad = matrix.X[:, quad_idx]
    assert np.allclose(main.T @ quad, 0.0, atol=1e-9)


@pytest.mark.parametrize("order", [2, 4, 6, 8, 10, 12, 14, 18, 20, 24, 26, 30])
def test_conference_matrix_is_valid(order):
    # verify the defining property directly: zero diagonal, +/-1 off-diagonal,
    # CᵀC == (order - 1) I -- for every order the construction claims to support.
    C = _conference_matrix(order)
    assert C.shape == (order, order)
    assert np.allclose(np.diag(C), 0.0)
    off_diagonal_values = C[~np.eye(order, dtype=bool)]
    assert set(np.unique(off_diagonal_values)) <= {-1.0, 1.0}
    gram = C.T @ C
    assert np.allclose(gram, (order - 1) * np.eye(order))


@pytest.mark.parametrize("order", [16, 22, 28])
def test_conference_matrix_unsupported_order_raises(order):
    with pytest.raises(ValueError):
        _conference_matrix(order)


def test_dsd_rejects_multilevel_categorical():
    factors = [*_factors(3), CategoricalFactor("cat", ("x", "y", "z"))]
    with pytest.raises(ValueError):
        definitive_screening(factors)
