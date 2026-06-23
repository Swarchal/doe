import pytest

from doe.analysis.model import build_model_matrix
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.factorial import full_factorial


def test_model_matrix_rejects_categorical_factors_until_phase_2_contrasts():
    design = full_factorial([CategoricalFactor("catalyst", ("A", "B"))])

    with pytest.raises(NotImplementedError, match="deferred to Phase 2"):
        build_model_matrix(design)


def test_quadratic_drops_squares_for_two_level_factors():
    # a pure +/-1 factor has x^2 == 1 (collinear with the intercept), so a quadratic
    # model must NOT emit a squared column for it -- otherwise X is rank-deficient.
    design = full_factorial([ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)])
    mm = build_model_matrix(design, order=2, interactions=True)
    assert not any(name.endswith("^2") for name in mm.term_names)


def test_quadratic_keeps_squares_for_three_level_factors():
    # a 3-level factor genuinely takes a value off {-1, +1}, so its square is estimable.
    design = full_factorial([ContinuousFactor("a", 0.0, 10.0)], levels=3)
    mm = build_model_matrix(design, order=2, interactions=True)
    assert "a^2" in mm.term_names
