"""Definitive screening designs (Phase 5a).

Written test-first to document the intended API and DSD defining properties.
Correctness anchors follow Jones & Nachtsheim (2011): 2k+1 runs, three coded levels,
main-effect orthogonality (including orthogonality to second-order terms), one all-zero
center run with two zeros per factor column, and the tabulated k=6 design.
"""

import numpy as np
import pytest

from doe.analysis.model import build_model_matrix
from doe.factors import CategoricalFactor, ContinuousFactor, MixtureFactor
from doe.generators.screening import (
    _conference_matrix,
    _jacobsthal_matrix,
    _order_exists,
    _skew_conference_matrix,
    _suggest_fake_factors,
    definitive_screening,
)


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


def test_dsd_k6_main_effects_independent_of_all_second_order():
    # The defining efficiency property of a definitive screening design (Jones & Nachtsheim
    # 2011): main effects are completely independent not just of quadratic terms but of *every*
    # two-factor interaction as well -- so a first-order fit's main effects are unbiased by any
    # active second-order effect. The k=6 case is the pure conference-matrix DSD (no fake
    # factor), where this holds exactly. This is strictly stronger than the main-vs-quadratic
    # orthogonality that ``test_dsd_main_effects_orthogonal`` checks.
    design = definitive_screening(_factors(6))
    assert design.n_runs == 13

    matrix = build_model_matrix(design, order=2, interactions=True)
    factor_names = set(design.factors.names)
    main_idx = [i for i, name in enumerate(matrix.term_names) if name in factor_names]
    inter_idx = [i for i, name in enumerate(matrix.term_names) if ":" in name]
    quad_idx = [i for i, name in enumerate(matrix.term_names) if name.endswith("^2")]
    assert len(main_idx) == 6
    assert len(inter_idx) == 6 * 5 // 2  # all 15 two-factor interactions
    assert len(quad_idx) == 6

    main = matrix.X[:, main_idx]
    assert np.allclose(main.T @ matrix.X[:, inter_idx], 0.0, atol=1e-9)
    assert np.allclose(main.T @ matrix.X[:, quad_idx], 0.0, atol=1e-9)


@pytest.mark.parametrize("order", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32])
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


@pytest.mark.parametrize("order", [4, 8, 12, 16, 20, 24, 32])
def test_skew_conference_matrix_is_skew_and_valid(order):
    # the doubling construction (which is how order 16 is reached) needs a *skew* input,
    # so the skew builder must deliver Cᵀ == -C on top of the conference property.
    C = _skew_conference_matrix(order)
    assert np.allclose(C.T, -C)
    assert np.allclose(np.diag(C), 0.0)
    assert set(np.unique(C[~np.eye(order, dtype=bool)])) <= {-1.0, 1.0}
    assert np.allclose(C.T @ C, (order - 1) * np.eye(order))


def test_conference_matrix_order_22_raises_as_nonexistent():
    # 22 = 2 mod 4 forces a symmetric conference matrix, which needs 21 to be a sum of two
    # squares -- it is not, so order 22 is impossible, not merely unimplemented.
    assert not _order_exists(22)
    with pytest.raises(ValueError, match="does not exist|no conference matrix of order 22 exists"):
        _conference_matrix(22)


def test_gf_cubic_jacobsthal_matrix_is_valid():
    # GF(27) is the first field needing the general polynomial path; it is what unlocks order 28
    Q = _jacobsthal_matrix(3, 3)
    assert Q.shape == (27, 27)
    assert np.allclose(np.diag(Q), 0.0)
    assert np.allclose(Q @ Q.T, 27 * np.eye(27) - np.ones((27, 27)))


def test_dsd_rejects_multilevel_categorical():
    factors = [*_factors(3), CategoricalFactor("cat", ("x", "y", "z"))]
    with pytest.raises(ValueError):
        definitive_screening(factors)


def test_dsd_rejects_mixture_with_own_message():
    # mixture rejection must not leak factorial._require_box_factors' "factorial designs" text
    factors = [MixtureFactor(chr(ord("a") + i)) for i in range(3)]
    with pytest.raises(ValueError, match="simplex"):
        definitive_screening(factors)


@pytest.mark.parametrize("k", [16, 28])
def test_dsd_exact_run_count_for_doubling_and_cubic_orders(k):
    # order 16 (doubling a skew order-8) and order 28 (Paley over GF(27)) are constructible, so
    # these factor counts get the exact 2k + 1 design with no fake factors
    design = definitive_screening(_factors(k))
    assert design.meta["fake_factors"] == 0
    assert design.n_runs == 2 * k + 1
    assert design.factors.names == [chr(ord("a") + i) for i in range(k)]
    coded = design.coded().to_numpy()
    assert set(np.unique(coded)) == {-1.0, 0.0, 1.0}
    # main effects stay orthogonal, which is the whole point of building at the exact order
    main = coded[: 2 * k]
    assert np.allclose(main.T @ main, np.diag(np.diag(main.T @ main)))


def test_dsd_odd_k_adds_one_fake_factor_to_reach_order_16():
    # k=15 previously skipped past order 16 (unconstructible) to order 18; now it lands on 16
    design = definitive_screening(_factors(15))
    assert design.meta["fake_factors"] == 1
    assert design.n_runs == 2 * 16 + 1
    assert len(design.factors.names) == 15


def test_dsd_auto_pads_past_the_nonexistent_order_22():
    # order 22 does not exist, so k=21 (odd) must advance to order 24 -- three fake factors
    design = definitive_screening(_factors(21))
    assert design.meta["fake_factors"] == 3
    assert design.n_runs == 2 * 24 + 1
    assert len(design.factors.names) == 21


def test_dsd_explicit_unconstructible_fake_factors_raises_actionable():
    # forcing order 22 must name k, say the matrix does not exist, and suggest a working count
    with pytest.raises(ValueError, match=r"k=21") as exc:
        definitive_screening(_factors(21), fake_factors=1)
    message = str(exc.value)
    assert "no conference matrix of order 22 exists" in message
    assert "sum of two squares" in message
    assert "fake_factors=3" in message


def test_dsd_explicit_odd_order_raises_actionable():
    # k=6 with one fake factor -> odd order 7; message flags parity and suggests even fake counts
    with pytest.raises(ValueError, match=r"even") as exc:
        definitive_screening(_factors(6), fake_factors=1)
    assert "fake_factors=0" in str(exc.value)


def test_suggest_fake_factors_parity_and_constructibility():
    # k=16 is now constructible outright -> no fake factors needed
    assert _suggest_fake_factors(16)[0] == 0
    # k=21 (odd): fake counts must be odd, and order 22 does not exist, so the first is 3
    assert all(nf % 2 == 1 for nf in _suggest_fake_factors(21))
    assert _suggest_fake_factors(21)[0] == 3


# --- Categorical DSD extension (Jones & Nachtsheim 2013, DSD-augment) ---------------------

# A frozen all-continuous k=4 DSD (coded), asserted bit-identical so the categorical extension
# never perturbs the continuous path.
_FROZEN_CONTINUOUS_K4 = np.array(
    [
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, -1.0, 1.0],
        [1.0, 1.0, 0.0, -1.0],
        [1.0, -1.0, 1.0, 0.0],
        [0.0, -1.0, -1.0, -1.0],
        [-1.0, 0.0, 1.0, -1.0],
        [-1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)


def _mixed(m, c, low=0.0, high=10.0):
    """m continuous + c two-level categorical factors, in that order."""
    return [
        *[ContinuousFactor(f"x{i}", low=low, high=high) for i in range(m)],
        *[CategoricalFactor(f"c{j}", ("L", "H")) for j in range(c)],
    ]


def _numeric_coded(design):
    """Numeric coded coordinates for every factor (categoricals -> +/-1), as an array."""
    from doe.analysis.model import coded_design_points

    return coded_design_points(design)


def test_continuous_path_bit_identical_to_frozen():
    # the all-continuous branch must be untouched by the categorical extension
    design = definitive_screening(_factors(4))
    assert np.allclose(_numeric_coded(design), _FROZEN_CONTINUOUS_K4)
    assert set(design.point_types) == {"dsd", "center"}


@pytest.mark.parametrize(
    "m,c,expected",
    # Table 4 (Jones & Nachtsheim 2013) n_DSD column, via conference-matrix order padding.
    [(4, 1, 14), (4, 2, 14), (4, 3, 18), (4, 4, 18), (5, 1, 14), (6, 1, 18)],
)
def test_categorical_dsd_run_counts_match_table_4(m, c, expected):
    design = definitive_screening(_mixed(m, c))
    assert design.n_runs == expected
    # n = 2*order + 2 (structural fold-over + one pseudo-center pair)
    assert design.point_types.count("pseudo-center") == 2
    assert design.point_types.count("dsd") == expected - 2


def test_categorical_dsd_information_matrix_matches_paper_eq2():
    # m=4, c=2 first-order XᵀX: diag (14,10,10,10,10,14,14), off-diagonals in {0, +/-2}
    # (Jones & Nachtsheim 2013, eq. 2). The determinant is the exhaustive-search maximum.
    design = definitive_screening(_mixed(4, 2))
    mm = build_model_matrix(design, order=1, interactions=False)
    xtx = mm.X.T @ mm.X
    assert np.allclose(np.diag(xtx), [14, 10, 10, 10, 10, 14, 14])
    off = xtx - np.diag(np.diag(xtx))
    assert set(np.unique(np.round(off, 9))) <= {-2.0, 0.0, 2.0}


def test_categorical_dsd_is_definitive_alias_matrix_main_rows_zero():
    # The defining property: for the first-order model, the alias matrix
    # (X1ᵀX1)⁻¹ X1ᵀ X2 against every quadratic and two-factor-interaction column has all-zero
    # main-effect rows -- main effects stay unbiased by any active second-order effect (Table 3).
    design = definitive_screening(_mixed(4, 2))
    factor_names = set(design.factors.names)
    mm = build_model_matrix(design, order=2, interactions=True)
    names = mm.term_names

    def is_main(name):
        if ":" in name or name.endswith("^2"):
            return False
        return name in factor_names or name.split("[")[0] in factor_names

    main = [i for i, n in enumerate(names) if is_main(n)]
    second = [i for i, n in enumerate(names) if ":" in n or n.endswith("^2")]
    x1 = np.column_stack([np.ones(mm.X.shape[0]), *[mm.X[:, i] for i in main]])
    x2 = mm.X[:, second]
    alias = np.linalg.solve(x1.T @ x1, x1.T @ x2)
    # rows 1: are the main-effect rows (row 0 is the intercept, which *is* aliased -- Table 3)
    assert np.allclose(alias[1:], 0.0, atol=1e-9)
    assert np.abs(alias[0]).max() > 1e-6  # intercept aliasing is present and surfaced


def test_categorical_dsd_column_structure():
    # categorical columns are zero-free +/-1; each continuous column has exactly two structural
    # zeros (its fold-over diagonal pair); the whole design is fold-over pairs.
    design = definitive_screening(_mixed(4, 2))
    coded = _numeric_coded(design)
    names = design.factors.names
    n1 = design.point_types.count("dsd")
    for j, name in enumerate(names):
        col = coded[:, j]
        if name.startswith("c"):  # categorical
            assert np.all(np.abs(col) == 1.0)
        else:  # continuous: two zeros among the structural rows
            assert np.count_nonzero(col[:n1] == 0.0) == 2
    # fold-over: the multiset of rows equals the multiset of negated rows
    rows = sorted(map(tuple, np.round(coded + 0.0, 6).tolist()))
    neg = sorted(map(tuple, np.round(-coded + 0.0, 6).tolist()))
    assert rows == neg


def test_categorical_dsd_pseudo_center_survives_round_trips():
    from doe.design import Design

    design = definitive_screening(_mixed(4, 2))
    assert "pseudo-center" in design.point_types
    assert "center" not in design.point_types  # not replicates -> not "center"

    # replicate / randomize / project / to_dict-from_dict all carry the tag
    assert design.replicate(2).point_types.count("pseudo-center") == 4
    assert set(design.randomize(seed=0).point_types) == set(design.point_types)
    projected = design.project(["x0", "x1", "c0"])
    assert "pseudo-center" in projected.point_types
    restored = Design.from_dict(design.to_dict())
    assert restored.point_types == design.point_types


def test_categorical_dsd_recovers_injected_response():
    # inject continuous mains + one quadratic + one categorical main; recover via the documented
    # fit-reduced-model flow (fit only the active terms, where residual dof is positive).
    from doe.analysis.fit import fit_ols

    design = definitive_screening(_mixed(3, 1))
    coded = _numeric_coded(design)
    x0, x1, cat = coded[:, 0], coded[:, 1], coded[:, 3]
    y = 20 + 3 * x0 - 2 * x1 + 1.5 * (x0**2) + 4 * cat
    measured = design.with_response("y", y)

    # reduced quadratic in the active continuous factors + the categorical main effect
    fit = fit_ols(measured, "y", order=2, interactions=False)
    summary = fit.summary()
    assert summary.loc["x0", "coefficient"] == pytest.approx(3.0, abs=1e-6)
    assert summary.loc["x1", "coefficient"] == pytest.approx(-2.0, abs=1e-6)
    assert summary.loc["x0^2", "coefficient"] == pytest.approx(1.5, abs=1e-6)
    assert summary.loc["c0[H]", "coefficient"] == pytest.approx(4.0, abs=1e-6)


def test_categorical_dsd_rejects_multilevel_but_accepts_two_level():
    # two-level categorical is fine; a three-level one still routes to d_optimal
    definitive_screening(_mixed(3, 1))  # no raise
    with pytest.raises(ValueError, match="two-level|d_optimal"):
        definitive_screening([*_factors(3), CategoricalFactor("cat", ("x", "y", "z"))])


def test_categorical_dsd_too_many_categoricals_raises():
    with pytest.raises(ValueError, match="d_optimal"):
        definitive_screening(_mixed(2, 11))


def test_categorical_dsd_extra_center_runs_add_pairs():
    design = definitive_screening(_mixed(4, 2), extra_center_runs=2)
    # base pseudo-center pair + 2 replicated pairs = 6 pseudo-center runs
    assert design.point_types.count("pseudo-center") == 6
    assert design.meta["categorical_signs"]["z"]  # signs recorded
