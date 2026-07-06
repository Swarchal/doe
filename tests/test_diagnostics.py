"""Phase 3a: design diagnostics (information matrix, VIF, leverage, correlation, efficiency)."""

import numpy as np
import pytest

from doe.analysis import diagnostics
from doe.analysis.model import build_model_matrix
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.factorial import full_factorial, plackett_burman
from doe.generators.optimal import candidate_grid


def _factors(n):
    return [ContinuousFactor(chr(ord("a") + i), low=-1.0, high=1.0) for i in range(n)]


def _model_matrix(design, *, order=1, interactions=True):
    mm = build_model_matrix(design, order=order, interactions=interactions)
    return mm.X, mm.term_names


# --------------------------------------------------------------------------- #
# Information-matrix scalars
# --------------------------------------------------------------------------- #


def test_information_matrix_is_xtx():
    x, _ = _model_matrix(full_factorial(_factors(3)))
    info = diagnostics.information_matrix(x)
    assert np.allclose(info, x.T @ x)


def test_orthogonal_design_has_unit_condition_number():
    # a 2^3 factorial with first-order + interaction terms has orthogonal columns
    x, _ = _model_matrix(full_factorial(_factors(3)))
    assert diagnostics.condition_number(x) == pytest.approx(1.0)


def test_log_det_information_matches_slogdet():
    x, _ = _model_matrix(full_factorial(_factors(3)))
    sign, logdet = np.linalg.slogdet(x.T @ x)
    assert sign > 0
    assert diagnostics.log_det_information(x) == pytest.approx(logdet)


def test_log_det_information_returns_negative_infinity_for_singular_matrix():
    x = np.ones((4, 2))
    assert diagnostics.log_det_information(x) == float("-inf")


# --------------------------------------------------------------------------- #
# VIF
# --------------------------------------------------------------------------- #


def test_orthogonal_design_has_unit_vif_for_every_term():
    x, names = _model_matrix(full_factorial(_factors(3)))
    v = diagnostics.vif(x, term_names=names)
    # one VIF per non-intercept term, all equal to 1 for an orthogonal design
    assert "Intercept" not in v.index
    assert np.allclose(v.to_numpy(), 1.0)


def test_vif_rejects_mismatched_term_names():
    x, names = _model_matrix(full_factorial(_factors(2)))
    with pytest.raises(ValueError, match="term_names"):
        diagnostics.vif(x, term_names=names[:-1])


# --------------------------------------------------------------------------- #
# Leverage
# --------------------------------------------------------------------------- #


def test_leverage_sums_to_number_of_terms():
    x, names = _model_matrix(full_factorial(_factors(3)))
    h = diagnostics.leverage(x)
    assert h.shape == (x.shape[0],)
    assert h.sum() == pytest.approx(len(names))


def test_saturated_design_has_a_run_at_unit_leverage():
    # 2^2 factorial fit with a saturated model (4 runs, 4 terms) -> every h == 1
    x, _ = _model_matrix(full_factorial(_factors(2)))
    h = diagnostics.leverage(x)
    assert np.isclose(h.max(), 1.0)


# --------------------------------------------------------------------------- #
# Correlation / alias matrix
# --------------------------------------------------------------------------- #


def test_orthogonal_design_correlation_is_identity():
    x, names = _model_matrix(full_factorial(_factors(3)))
    corr = diagnostics.correlation_matrix(x, names)
    # intercept dropped; remaining terms mutually uncorrelated
    assert "Intercept" not in corr.columns
    assert np.allclose(corr.to_numpy(), np.eye(corr.shape[0]), atol=1e-9)


def test_correlation_matrix_rejects_mismatched_term_names():
    x, names = _model_matrix(full_factorial(_factors(2)))
    with pytest.raises(ValueError, match="term_names"):
        diagnostics.correlation_matrix(x, names[:-1])


def test_plackett_burman_partial_aliasing_is_one_third():
    # a PB main effect leaks +/- 1/3 into two-factor interactions (the known PB alias)
    design = plackett_burman(_factors(11))
    x, names = _model_matrix(design, order=1, interactions=True)
    corr = diagnostics.correlation_matrix(x, names)
    main = [n for n in corr.columns if ":" not in n]
    inter = [n for n in corr.columns if ":" in n]
    block = corr.loc[main, inter].to_numpy()
    nonzero = np.abs(block[np.abs(block) > 1e-9])
    assert np.allclose(nonzero, 1.0 / 3.0)


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #


def test_full_factorial_is_d_efficient():
    design = full_factorial(_factors(3))
    eff = diagnostics.efficiency(design, order=1, interactions=True)
    assert isinstance(eff, diagnostics.Efficiency)
    assert eff.d == pytest.approx(1.0)


def test_efficiency_returns_zeroes_for_singular_design():
    design = full_factorial(_factors(2))
    duplicated = full_factorial(_factors(2))
    duplicated.runs = design.runs.iloc[[0, 0, 0, 0]].reset_index(drop=True)

    eff = diagnostics.efficiency(duplicated, order=1, interactions=True)

    assert eff == diagnostics.Efficiency(d=0.0, a=0.0, g=0.0, i=0.0)


def test_efficiencies_are_in_unit_interval():
    design = full_factorial(_factors(3))
    eff = diagnostics.efficiency(design, order=1, interactions=True)
    for value in (eff.d, eff.a, eff.g, eff.i):
        assert 0.0 <= value <= 1.0 + 1e-9


def test_efficiency_supports_mixed_categorical_region():
    factors = [
        ContinuousFactor("temp", 0.0, 100.0),
        CategoricalFactor("catalyst", ("A", "B", "C")),
    ]
    design = full_factorial(factors, levels=3)
    region = candidate_grid(factors, levels=3)

    eff = diagnostics.efficiency(design, order=2, interactions=True, region=region)

    assert eff.d > 0.0
    assert 0.0 <= eff.i <= 1.0 + 1e-9
