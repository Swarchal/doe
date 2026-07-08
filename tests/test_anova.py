"""Quadratic fitting, inference, and ANOVA / lack-of-fit.

Written test-first to document the intended API and expected numbers. A face-centered
CCD supports the full second-order model, so we anchor against a known quadratic.
"""

import numpy as np

from doe.analysis.anova import (
    adjusted_r2,
    anova_table,
    lack_of_fit,
    predicted_r2,
    press,
)
from doe.analysis.fit import fit_ols
from doe.factors import ContinuousFactor
from doe.generators.factorial import full_factorial
from doe.generators.rsm import central_composite


def _ccd(center=5):
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    return central_composite(factors, center=center)


def _quadratic_response(design):
    """y = 50 + 3*x1 - 2*x2 + 4*x1^2 - 1*x1*x2  (built in coded units, no x2^2 term)."""
    coded = design.coded().to_numpy()
    x1, x2 = coded[:, 0], coded[:, 1]
    return 50.0 + 3.0 * x1 - 2.0 * x2 + 4.0 * x1**2 - 1.0 * x1 * x2


# --------------------------------------------------------------------------- #
# Quadratic model + coefficient recovery
# --------------------------------------------------------------------------- #


def test_quadratic_model_convenience_matches_explicit_flags():
    design = _ccd()
    y = _quadratic_response(design)
    explicit = fit_ols(design, y, order=2, interactions=True)
    convenience = fit_ols(design, y, model="quadratic")
    assert convenience.term_names == explicit.term_names
    assert np.allclose(convenience.coefficients, explicit.coefficients)


def test_fit_recovers_known_quadratic_coefficients():
    design = _ccd()
    y = _quadratic_response(design)
    result = fit_ols(design, y, model="quadratic")
    coef = result.summary()["coefficient"]  # term -> coefficient

    assert np.isclose(coef["Intercept"], 50.0)
    assert np.isclose(coef["a"], 3.0)
    assert np.isclose(coef["b"], -2.0)
    assert np.isclose(coef["a:b"], -1.0)
    assert np.isclose(coef["a^2"], 4.0)
    assert np.isclose(coef["b^2"], 0.0, atol=1e-8)
    assert np.isclose(result.r_squared, 1.0)


# --------------------------------------------------------------------------- #
# FitResult inference fields
# --------------------------------------------------------------------------- #


def test_fitresult_residual_dof():
    design = _ccd(center=5)  # 4 + 4 + 5 = 13 runs; quadratic has 6 terms
    result = fit_ols(design, _quadratic_response(design), model="quadratic")
    assert result.dof_resid == design.n_runs - len(result.term_names)
    assert result.dof_resid == 13 - 6


def test_fitresult_standard_errors_positive_with_noise():
    design = _ccd(center=5)
    rng = np.random.default_rng(0)
    y = _quadratic_response(design) + rng.normal(scale=1.0, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")

    assert result.std_errors.shape == result.coefficients.shape
    assert np.all(np.isfinite(result.std_errors))
    assert np.all(result.std_errors > 0.0)
    assert result.cov_beta.shape == (len(result.term_names), len(result.term_names))
    assert result.mse > 0.0


def test_fitresult_conf_int_brackets_coefficients():
    design = _ccd(center=5)
    rng = np.random.default_rng(1)
    y = _quadratic_response(design) + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")

    ci = result.conf_int(level=0.95)
    assert list(ci.index) == result.term_names
    assert list(ci.columns) == ["lower", "upper"]
    assert np.all(ci["lower"].to_numpy() <= result.coefficients)
    assert np.all(result.coefficients <= ci["upper"].to_numpy())


def test_fitresult_pvalues_in_unit_interval():
    design = _ccd(center=5)
    rng = np.random.default_rng(2)
    y = _quadratic_response(design) + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")
    assert np.all(result.p_values >= 0.0)
    assert np.all(result.p_values <= 1.0)


# --------------------------------------------------------------------------- #
# ANOVA table (sequential / Type I)
# --------------------------------------------------------------------------- #


def test_anova_table_shape():
    design = _ccd(center=5)
    rng = np.random.default_rng(3)
    y = _quadratic_response(design) + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")

    table = anova_table(result, design, y)
    assert {"SS", "df", "MS", "F", "p"}.issubset(table.columns)
    assert "Residual" in table.index
    assert "Total" in table.index


def test_anova_sequential_ss_partitions_total():
    design = _ccd(center=5)
    rng = np.random.default_rng(4)
    y = _quadratic_response(design) + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")
    table = anova_table(result, design, y)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    term_rows = table.drop(index=["Residual", "Total"])
    model_ss = float(term_rows["SS"].sum())
    ss_resid = float(result.residuals @ result.residuals)

    # sequential term SS plus residual SS reconstruct the total corrected SS
    assert np.isclose(model_ss + ss_resid, ss_tot)
    assert np.isclose(float(table.loc["Total", "SS"]), ss_tot)
    # degrees of freedom add up too
    assert int(table.loc["Total", "df"]) == design.n_runs - 1
    assert int(term_rows["df"].sum()) + int(table.loc["Residual", "df"]) == design.n_runs - 1


# --------------------------------------------------------------------------- #
# Lack of fit (pure error from replicated center points)
# --------------------------------------------------------------------------- #


def test_lack_of_fit_pure_error_dof():
    design = _ccd(center=5)
    rng = np.random.default_rng(5)
    y = _quadratic_response(design) + rng.normal(scale=0.3, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")
    lof = lack_of_fit(result, design, y)
    assert lof.df_pe == design.n_center - 1  # 4


def test_lack_of_fit_not_significant_for_true_quadratic():
    design = _ccd(center=5)
    rng = np.random.default_rng(6)
    y = _quadratic_response(design) + rng.normal(scale=0.2, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")
    lof = lack_of_fit(result, design, y)
    # the quadratic model is adequate -> lack-of-fit should not be significant
    assert lof.p_value > 0.05


def test_lack_of_fit_pools_pure_error_from_non_center_replicates():
    # a replicated 2^2 factorial has no center points, but every corner is run twice, so pure
    # error is still estimable (4 groups of 2 -> df_pe = 4). The old center-only rule rejected
    # this; pooling over identical settings now uses those replicates.
    design = full_factorial([ContinuousFactor("a", 0.0, 1.0), ContinuousFactor("b", 0.0, 1.0)])
    design = design.replicate(2)
    assert design.n_center == 0
    rng = np.random.default_rng(11)
    coded = design.coded().to_numpy()
    y = 5.0 + 2.0 * coded[:, 0] - coded[:, 1] + rng.normal(scale=0.1, size=design.n_runs)
    result = fit_ols(design, y, order=1, interactions=False)  # 3 terms, dof_resid = 5
    lof = lack_of_fit(result, design, y)
    assert lof.df_pe == 4  # four corners, each replicated once
    assert lof.df_lof == result.dof_resid - lof.df_pe
    assert 0.0 <= lof.p_value <= 1.0


def test_lack_of_fit_significant_for_linear_fit_to_curved_data():
    design = _ccd(center=5)
    coded = design.coded().to_numpy()
    x1, x2 = coded[:, 0], coded[:, 1]
    rng = np.random.default_rng(7)
    # strong curvature; tiny noise gives the center points a small pure-error spread
    y = 50.0 + 6.0 * x1**2 + 5.0 * x2**2 + rng.normal(scale=0.05, size=design.n_runs)
    result = fit_ols(design, y, order=1, interactions=True)  # linear: misses the curvature
    lof = lack_of_fit(result, design, y)
    assert lof.p_value < 0.05


# --------------------------------------------------------------------------- #
# PRESS / predicted R-squared
# --------------------------------------------------------------------------- #


def test_press_and_predicted_r2():
    design = _ccd(center=5)
    rng = np.random.default_rng(8)
    y = _quadratic_response(design) + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")

    assert press(result) > 0.0
    # predicted R^2 (leave-one-out) is always below the in-sample R^2
    assert predicted_r2(result) < result.r_squared
    # adjusted R^2 likewise penalises for parameters
    assert adjusted_r2(result) <= result.r_squared
