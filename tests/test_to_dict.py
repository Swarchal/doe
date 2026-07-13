"""``to_dict`` serialization for analysis/optimization results (doe-service Milestone 0).

Every response body the web service returns is one of these ``to_dict``s -- see
``docs/WEBSERVICE_API.md`` for the wire contract and ``docs/WEBSERVICE_BUILD.md`` section
0 for the build plan. Tests reuse the existing golden anchors (known injected effects, a
face-centered CCD's known quadratic) rather than inventing new numbers, matching how the
rest of the suite anchors correctness.
"""

import json

import numpy as np
import pytest

from doe.analysis import diagnostics
from doe.analysis.anova import LackOfFit, anova_records, lack_of_fit
from doe.analysis.fit import FitResult, SaturatedFitWarning, fit_ols
from doe.analysis.optimize import ResponseGoal, desirability, optimum, stationary_point
from doe.factors import ContinuousFactor, MixtureFactor
from doe.generators.factorial import full_factorial
from doe.generators.mixture import simplex_lattice
from doe.generators.rsm import central_composite


def _ccd(center=5):
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    return central_composite(factors, center=center)


def _quadratic_response(design):
    """y = 50 + 3*x1 - 2*x2 + 4*x1^2 - 1*x1*x2 (same anchor as test_anova.py)."""
    coded = design.coded().to_numpy()
    x1, x2 = coded[:, 0], coded[:, 1]
    return 50.0 + 3.0 * x1 - 2.0 * x2 + 4.0 * x1**2 - 1.0 * x1 * x2


# --------------------------------------------------------------------------- #
# FitResult.to_dict
# --------------------------------------------------------------------------- #


def test_fitresult_to_dict_terms_carry_known_injected_effects():
    # same known-effects anchor as test_fit.py::test_fit_recovers_known_effects
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2)
    coded = design.coded().to_numpy()
    a, b = coded[:, 0], coded[:, 1]
    y = 10 + 3 * a + 2 * b + 1.5 * a * b

    result = fit_ols(design, y, order=1, interactions=True)
    payload = result.to_dict()

    terms = {row["term"]: row for row in payload["terms"]}
    assert terms["Intercept"]["coefficient"] == pytest.approx(10.0)
    assert terms["a"]["effect"] == pytest.approx(6.0)
    assert terms["b"]["effect"] == pytest.approx(4.0)
    assert terms["a:b"]["effect"] == pytest.approx(3.0)
    assert payload["r_squared"] == pytest.approx(1.0)
    assert payload["model"] == {"order": 1, "interactions": True}
    assert payload["dof_resid"] == result.dof_resid


def test_fitresult_to_dict_matches_summary_and_conf_int():
    design = _ccd(center=5)
    rng = np.random.default_rng(0)
    y = _quadratic_response(design) + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")

    payload = result.to_dict(confidence=0.9)
    summary = result.summary()
    ci = result.conf_int(0.9)
    for row in payload["terms"]:
        term = row["term"]
        assert row["coefficient"] == pytest.approx(summary.loc[term, "coefficient"])
        assert row["std_error"] == pytest.approx(summary.loc[term, "std_error"])
        assert row["ci_low"] == pytest.approx(ci.loc[term, "lower"])
        assert row["ci_high"] == pytest.approx(ci.loc[term, "upper"])
    assert payload["mse"] == pytest.approx(result.mse)
    assert payload["fitted"] == pytest.approx(result.fitted.tolist())
    assert payload["residuals"] == pytest.approx(result.residuals.tolist())


def test_fitresult_to_dict_saturated_serializes_null_std_error_and_warns():
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    design = full_factorial(factors, levels=2)  # 4 runs
    coded = design.coded().to_numpy()
    y = 10 + 3 * coded[:, 0] + 2 * coded[:, 1] + 1.5 * coded[:, 0] * coded[:, 1]  # 4 terms

    with pytest.warns(SaturatedFitWarning):
        result = fit_ols(design, y, model="linear")  # saturated: dof_resid == 0
    assert isinstance(result, FitResult)

    payload = result.to_dict()
    assert all(row["std_error"] is None for row in payload["terms"])
    assert all(row["t"] is None for row in payload["terms"])
    assert all(row["ci_low"] is None for row in payload["terms"])
    assert payload["mse"] is None
    assert payload["adjusted_r2"] is None

    # SaturatedFitWarning must still satisfy the pre-existing bare-UserWarning tests
    with pytest.warns(UserWarning):
        fit_ols(design, y, model="linear")


def test_fitresult_to_dict_scheffe_serializes_null_effect():
    # simplex_lattice({3, 2}) is 6 runs against a 6-term scheffe-quadratic model (3 linear
    # + 3 cross terms) -- saturated, same as test_fit.py::test_predict_honors_scheffe_mixture_path
    components = [MixtureFactor("x1"), MixtureFactor("x2"), MixtureFactor("x3")]
    design = simplex_lattice(components, degree=2)
    coded = design.coded().to_numpy()
    x1, x2, x3 = coded[:, 0], coded[:, 1], coded[:, 2]
    y = 3 * x1 + 5 * x2 + 2 * x3 + 4 * x1 * x2 - x1 * x3 + 2 * x2 * x3

    with pytest.warns(SaturatedFitWarning):
        result = fit_ols(design, y, model="scheffe-quadratic")
    payload = result.to_dict()

    assert all(row["effect"] is None for row in payload["terms"])
    # coefficients are still real numbers -- only the +/-1-swing "effect" is undefined
    assert all(row["coefficient"] is not None for row in payload["terms"])
    json.dumps(payload)


# --------------------------------------------------------------------------- #
# anova_records / LackOfFit.to_dict
# --------------------------------------------------------------------------- #


def test_anova_records_matches_anova_table():
    design = _ccd(center=5)
    rng = np.random.default_rng(3)
    y = _quadratic_response(design) + rng.normal(scale=0.5, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")

    table = result.anova()
    records = anova_records(result)

    assert [r["term"] for r in records] == list(table.index)
    assert records[-1]["term"] == "Total"
    assert records[-2]["term"] == "Residual"
    # Residual/Total rows have undefined F/p -> null, not NaN
    assert records[-1]["f"] is None and records[-1]["p"] is None
    for row, (_term, series) in zip(records, table.iterrows(), strict=True):
        assert row["ss"] == pytest.approx(series["SS"])
        assert row["df"] == pytest.approx(series["df"])
    json.dumps(records)


def test_lack_of_fit_to_dict_field_names():
    design = _ccd(center=5)
    rng = np.random.default_rng(5)
    y = _quadratic_response(design) + rng.normal(scale=0.3, size=design.n_runs)
    result = fit_ols(design, y, model="quadratic")
    lof = lack_of_fit(result, design, y)

    payload = lof.to_dict()
    assert isinstance(lof, LackOfFit)
    assert payload == {
        "ss_lof": pytest.approx(lof.ss_lof),
        "df_lof": lof.df_lof,
        "ss_pe": pytest.approx(lof.ss_pe),
        "df_pe": lof.df_pe,
        "f": pytest.approx(lof.f_stat),
        "p": pytest.approx(lof.p_value),
    }
    json.dumps(payload)


# --------------------------------------------------------------------------- #
# Efficiency.to_dict
# --------------------------------------------------------------------------- #


def test_efficiency_to_dict():
    factors = [ContinuousFactor(chr(ord("a") + i), low=-1.0, high=1.0) for i in range(3)]
    design = full_factorial(factors)
    eff = diagnostics.efficiency(design, order=1, interactions=True)

    payload = eff.to_dict()
    assert payload == {
        "d": pytest.approx(eff.d),
        "a": pytest.approx(eff.a),
        "g": pytest.approx(eff.g),
        "i": pytest.approx(eff.i),
    }
    assert payload["d"] == pytest.approx(1.0)
    json.dumps(payload)


# --------------------------------------------------------------------------- #
# StationaryPoint / Optimum / DesirabilityResult .to_dict
# --------------------------------------------------------------------------- #


def _surface_fit(coded_response, *, response_name=None):
    design = _ccd()
    coded = design.coded().to_numpy()
    y = coded_response(coded[:, 0], coded[:, 1])
    if response_name is not None:
        design = design.with_response(response_name, y)
        return fit_ols(design, response_name, model="quadratic")
    return fit_ols(design, y, model="quadratic")


def test_stationary_point_to_dict_includes_eigenvectors():
    result = _surface_fit(
        lambda x1, x2: 50.0 + 3.0 * x1 + 2.0 * x2 - 4.0 * x1**2 - 3.0 * x2**2 - 1.0 * x1 * x2,
        response_name="yield_pct",
    )
    sp = stationary_point(result)
    payload = sp.to_dict()

    assert payload["kind"] == "maximum"
    assert payload["response_name"] == "yield_pct"
    assert payload["eigenvalues"] == pytest.approx(sp.eigenvalues.tolist())
    assert np.allclose(payload["eigenvectors"], sp.eigenvectors)
    assert payload["coded"] == pytest.approx(sp.coded.tolist())
    assert payload["natural"]["a"] == pytest.approx(sp.natural["a"])
    json.dumps(payload)


def test_optimum_to_dict():
    result = _surface_fit(lambda x1, x2: 50.0 + 10.0 * x1 + 2.0 * x2 - x1**2 - x2**2)
    opt = optimum(result, maximize=True)
    payload = opt.to_dict()

    assert payload["maximize"] is True
    assert payload["at_bound"] is True
    assert payload["response_name"] is None
    assert payload["coded"] == pytest.approx(opt.coded.tolist())
    json.dumps(payload)


def test_desirability_to_dict():
    design = _ccd()
    coded = design.coded().to_numpy()
    yield_design = design.with_response("yield_pct", 50.0 + 10.0 * coded[:, 0])
    impurity_design = design.with_response("impurity_pct", 50.0 - 10.0 * coded[:, 0])
    r1 = fit_ols(yield_design, "yield_pct", order=1, interactions=True)
    r2 = fit_ols(impurity_design, "impurity_pct", order=1, interactions=True)

    des = desirability(
        [
            ResponseGoal(r1, goal="max", low=50.0, high=60.0),
            ResponseGoal(r2, goal="min", low=40.0, high=50.0),
        ]
    )
    payload = des.to_dict()

    assert payload["overall"] == pytest.approx(des.overall)
    assert payload["responses"] == pytest.approx(des.responses.to_dict())
    assert payload["individual"] == pytest.approx(des.individual.to_dict())
    assert set(payload["responses"]) == {"yield_pct", "impurity_pct"}
    json.dumps(payload)


# --------------------------------------------------------------------------- #
# json.dumps succeeds on every new to_dict -- the "JSON has no NaN" test
# --------------------------------------------------------------------------- #


def test_all_new_to_dicts_are_json_dumpable():
    design = _ccd(center=5)
    y = _quadratic_response(design)
    result = fit_ols(design, y, model="quadratic")

    json.dumps(result.to_dict())
    json.dumps(anova_records(result))
    json.dumps(lack_of_fit(result, design, y).to_dict())
    json.dumps(diagnostics.efficiency(design, order=2, interactions=True).to_dict())
    json.dumps(stationary_point(result).to_dict())
    json.dumps(optimum(result, maximize=True).to_dict())
    goal = ResponseGoal(result, goal="max", low=float(y.min()), high=float(y.max()))
    json.dumps(desirability([goal]).to_dict())
