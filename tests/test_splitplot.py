"""Split-plot designs and the two-stratum REML fit (Phase 5b).

Anchors: the whole-plot (hard-to-change) factor is constant within each plot, the plot count is
``n_whole_plot_settings × reps``, randomization never splits a plot, and -- for the fit --
REML matches the closed-form balanced-split-plot variance components while ``fit_ols`` reports
an anticonservatively *small* whole-plot standard error (the classic split-plot trap).
"""

import numpy as np
import pytest

from doe.factors import CategoricalFactor, ContinuousFactor, FactorSet, MixtureFactor
from doe.generators.factorial import full_factorial
from doe.generators.splitplot import split_plot


def _factors():
    return [
        ContinuousFactor("oven", 200, 400, hard_to_change=True),  # whole-plot
        ContinuousFactor("time", 5, 15),  # sub-plot
        ContinuousFactor("conc", 1, 3),  # sub-plot
    ]


def test_split_plot_grouping_and_plot_count():
    d = split_plot(_factors())
    # whole-plot design = full factorial on 1 factor (2 settings); sub-plot = full on 2 (4 runs)
    assert d.n_whole_plots == 2
    assert d.n_runs == 2 * 4
    # the hard-to-change factor is constant within each plot; sub-plot factors vary
    for plot in range(d.n_whole_plots):
        idx = d.whole_plot_indices(plot)
        assert d.runs["oven"].to_numpy()[idx].std() == 0.0
        assert d.runs["time"].to_numpy()[idx].std() > 0.0


def test_split_plot_reps_make_new_plots():
    d = split_plot(_factors(), n_whole_plot_reps=3)
    assert d.n_whole_plots == 2 * 3
    assert d.n_runs == 2 * 3 * 4


def test_split_plot_composition_with_passed_design():
    # a split-plot fractional/RSM falls out by passing a ready-made component design
    sp = full_factorial([ContinuousFactor("time", 5, 15), ContinuousFactor("conc", 1, 3)], levels=3)
    d = split_plot(_factors(), sub_plot_design=sp)
    assert d.n_runs == 2 * sp.n_runs  # 2 whole plots, 9 sub-plot runs each
    for plot in range(d.n_whole_plots):
        idx = d.whole_plot_indices(plot)
        assert d.runs["oven"].to_numpy()[idx].std() == 0.0


def test_split_plot_seed_returns_randomized_and_never_splits():
    d = split_plot(_factors(), seed=7)
    assert d.meta["randomized"] is True
    from itertools import groupby

    lengths = [len(list(g)) for _, g in groupby(d.whole_plots)]
    assert lengths == [4, 4]  # plots stay contiguous
    for plot in set(d.whole_plots):
        idx = d.whole_plot_indices(plot)
        assert d.runs["oven"].to_numpy()[idx].std() == 0.0


def test_split_plot_requires_both_strata():
    with pytest.raises(ValueError, match="ordinary design"):
        split_plot([ContinuousFactor("a", 0, 1), ContinuousFactor("b", 0, 1)])  # no HTC factor
    with pytest.raises(ValueError, match="ordinary design"):
        split_plot(
            [ContinuousFactor("a", 0, 1, hard_to_change=True),
             ContinuousFactor("b", 0, 1, hard_to_change=True)]  # no sub-plot factor
        )


def test_split_plot_rejects_mixture():
    with pytest.raises(ValueError, match="mixture|simplex"):
        split_plot([MixtureFactor("a"), MixtureFactor("b"), MixtureFactor("c")])


def test_split_plot_component_design_factor_mismatch_raises():
    wrong = full_factorial([ContinuousFactor("nope", 0, 1)])
    with pytest.raises(ValueError, match="exactly those factors"):
        split_plot(_factors(), sub_plot_design=wrong)


def test_split_plot_categorical_whole_plot_factor():
    factors = [
        CategoricalFactor("lot", ("A", "B"), hard_to_change=True),
        ContinuousFactor("time", 5, 15),
    ]
    d = split_plot(factors)
    assert d.n_whole_plots == 2
    for plot in range(d.n_whole_plots):
        idx = d.whole_plot_indices(plot)
        assert len(set(d.runs["lot"].to_numpy()[idx])) == 1


def test_split_plot_keeps_full_factor_set():
    d = split_plot(_factors())
    assert isinstance(d.factors, FactorSet)
    assert [f.name for f in d.factors.whole_plot_factors] == ["oven"]
    # coded units are still [-1, +1] per factor (unlike the Scheffé path)
    assert set(np.unique(d.coded().to_numpy())) <= {-1.0, 1.0}


# --- REML variance components + fit_gls (Phase 5b) ------------------------------------------

import pandas as pd  # noqa: E402

from doe.analysis.fit import fit_gls, fit_ols  # noqa: E402
from doe.analysis.variance import reml_variance_components  # noqa: E402
from doe.design import Design  # noqa: E402


def _balanced_split_plot(r=4, seed=0):
    """A (WP, 2 levels) x r reps x B (SP, 2 levels); returns (design, y, whole_plots, A, B)."""
    A, B, wp = [], [], []
    plot = 0
    for a in (-1.0, 1.0):
        for _rep in range(r):
            for b in (-1.0, 1.0):
                A.append(a)
                B.append(b)
                wp.append(plot)
            plot += 1
    A = np.array(A)
    B = np.array(B)
    wp = tuple(wp)
    rng = np.random.default_rng(seed)
    wp_dev = rng.normal(0, 3, size=plot)
    sp_err = rng.normal(0, 1, size=len(A))
    y = 10 + 2 * A + 1.5 * B + 0.5 * A * B + wp_dev[list(wp)] + sp_err
    factors = FactorSet(
        [ContinuousFactor("A", -1, 1, hard_to_change=True), ContinuousFactor("B", -1, 1)]
    )
    design = Design(pd.DataFrame({"A": A, "B": B}), factors, whole_plots=wp).with_response("y", y)
    return design, y, wp, A, B


def test_reml_matches_closed_form_balanced_anova():
    # For a *balanced* split-plot, REML equals the ANOVA/method-of-moments variance components
    # exactly -- computed here independently, so there is no transcription risk.
    design, y, wp, A, B = _balanced_split_plot()
    from doe.analysis.model import build_model_matrix

    x = build_model_matrix(design, order=1, interactions=True).X
    sigma2_wp, sigma2, _ll = reml_variance_components(x, y, wp)

    n_plots = len(set(wp))
    n_sp = len(A) // n_plots
    plot_means = np.array([y[np.array(wp) == p].mean() for p in range(n_plots)])
    a_plot = np.array([A[np.array(wp) == p][0] for p in range(n_plots)])
    xwp = np.column_stack([np.ones(n_plots), a_plot])
    rss_wp = ((plot_means - xwp @ np.linalg.lstsq(xwp, plot_means, rcond=None)[0]) ** 2).sum()
    ms_wp = rss_wp / (n_plots - 2)
    within = y - plot_means[list(wp)]
    xsp = np.column_stack([B, A * B])
    ss_sp = ((within - xsp @ np.linalg.lstsq(xsp, within, rcond=None)[0]) ** 2).sum()
    ms_sp = ss_sp / (n_plots - 2)
    mm_sigma2 = ms_sp
    mm_sigma2_wp = ms_wp - ms_sp / n_sp

    assert sigma2 == pytest.approx(mm_sigma2, rel=1e-6)
    assert sigma2_wp == pytest.approx(mm_sigma2_wp, rel=1e-6)


def test_reml_recovers_injected_components_in_simulation():
    # seeded simulation-recovery: many whole plots, REML lands near the true (9, 1) components
    A, B, wp = [], [], []
    plot = 0
    for a in (-1.0, 1.0):
        for _rep in range(40):
            for b in (-1.0, 1.0):
                A.append(a)
                B.append(b)
                wp.append(plot)
            plot += 1
    A = np.array(A)
    B = np.array(B)
    rng = np.random.default_rng(11)
    wp_dev = rng.normal(0, 3.0, size=plot)  # sigma_wp = 3 -> sigma2_wp = 9
    sp_err = rng.normal(0, 1.0, size=len(A))
    y = 5 + A + B + wp_dev[list(wp)] + sp_err
    factors = FactorSet(
        [ContinuousFactor("A", -1, 1, hard_to_change=True), ContinuousFactor("B", -1, 1)]
    )
    design = Design(pd.DataFrame({"A": A, "B": B}), factors, whole_plots=tuple(wp))
    from doe.analysis.model import build_model_matrix

    x = build_model_matrix(design, order=1, interactions=True).X
    sigma2_wp, sigma2, _ = reml_variance_components(x, y, tuple(wp))
    assert sigma2 == pytest.approx(1.0, abs=0.25)
    assert sigma2_wp == pytest.approx(9.0, abs=3.0)


def test_ols_understates_whole_plot_standard_error_the_trap():
    # The classic split-plot trap: OLS pools the two strata and reports a *smaller* whole-plot
    # standard error than the correct GLS fit.
    design, *_ = _balanced_split_plot()
    gls = fit_gls(design, "y")
    ols = fit_ols(design, "y")
    se_gls_A = gls.summary().loc["A", "std_error"]
    se_ols_A = ols.summary().loc["A", "std_error"]
    assert se_ols_A < se_gls_A


def test_fit_gls_carries_variance_components_and_two_stratum_df():
    design, _y, wp, *_ = _balanced_split_plot()
    fit = fit_gls(design, "y")
    assert fit.sigma2_wp is not None and fit.sigma2_wp > 0
    assert fit.n_whole_plots == len(set(wp))
    dof = dict(zip(fit.term_names, fit.dof_terms, strict=True))
    # whole-plot-level terms (intercept + A) vs sub-plot terms (B, A:B) use different df
    assert dof["A"] == fit.n_whole_plots - 2
    assert dof["B"] == fit.dof_resid
    # effects keep the 2 x coefficient meaning (coded +/-1 factors)
    i = fit.term_names.index("B")
    assert fit.effects[i] == pytest.approx(2 * fit.coefficients[i])


def test_fit_gls_requires_whole_plots():
    factors = FactorSet([ContinuousFactor("A", -1, 1), ContinuousFactor("B", -1, 1)])
    design = Design(
        pd.DataFrame({"A": [-1.0, 1.0, -1.0, 1.0], "B": [-1.0, -1.0, 1.0, 1.0]}), factors
    ).with_response("y", [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="whole_plots"):
        fit_gls(design, "y")


def test_gls_fit_refuses_single_stratum_statistics():
    # A GLS result inherits the post-fit methods, all of which are built on ONE pooled residual
    # variance -- on a split-plot fit they would test whole-plot terms against the (much smaller)
    # sub-plot error, reintroducing the very trap fit_gls exists to remove. They must refuse.
    design, *_ = _balanced_split_plot()
    fit = fit_gls(design, "y")
    for call in (fit.anova, fit.press, fit.predicted_r2, fit.lack_of_fit):
        with pytest.raises(NotImplementedError):
            call()
    with pytest.raises(NotImplementedError):
        fit.predict({"A": 1.0, "B": 1.0}, interval="confidence")
    # point prediction has no error stratum in it, so it still works
    assert np.isfinite(fit.predict({"A": 1.0, "B": 1.0}))
    # and the two-stratum-aware statistics are unaffected
    assert np.isfinite(fit.conf_int().loc["A", "lower"])


def test_ols_on_the_same_design_still_offers_them():
    # guard against the refusal above leaking onto ordinary OLS results
    design, *_ = _balanced_split_plot()
    fit = fit_ols(design, "y")
    assert fit.anova().shape[0] > 0
    assert np.isfinite(fit.press())
