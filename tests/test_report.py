"""Tests for the self-contained HTML analysis report (``doe.report``)."""

from __future__ import annotations

import base64
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from doe import (
    ContinuousFactor,
    MixtureFactor,
    analysis_report,
    central_composite,
    fit_ols,
    full_factorial,
    simplex_lattice,
)


@pytest.fixture
def ccd_fits():
    rng = np.random.default_rng(0)
    design = central_composite(
        [ContinuousFactor("temp", 40, 80, units="C"), ContinuousFactor("time", 5, 15)],
        alpha="faced",
    )
    c = design.coded()
    y = 80 + 3 * c.temp - 2 * c.time - 4 * c.temp**2 + rng.normal(0, 0.5, design.n_runs)
    z = 2 + 0.5 * c.temp + rng.normal(0, 0.1, design.n_runs)
    measured = design.with_responses(yield_pct=y, impurity=z)
    return fit_ols(measured, "yield_pct", order=2), fit_ols(measured, "impurity", order=2)


def _embedded_pngs(html: str) -> list[bytes]:
    return [
        base64.b64decode(m) for m in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    ]


def test_report_is_self_contained_with_embedded_figures(ccd_fits):
    fit, _ = ccd_fits
    html = analysis_report(fit)

    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # no external assets: nothing fetched over the network
    assert "http://" not in html and "https://" not in html
    pngs = _embedded_pngs(html)
    assert len(pngs) >= 6  # pareto, half-normal, pred/actual, resid, QQ, leverage, contour
    assert all(p.startswith(b"\x89PNG") for p in pngs)
    for section in ["Coefficients", "ANOVA", "Lack of fit", "Design", "Runs"]:
        assert section in html
    assert "yield_pct" in html  # heading from the response name
    assert "temp" in html and "40 – 80 C" in html


def test_report_lists_every_term_and_run(ccd_fits):
    fit, _ = ccd_fits
    html = analysis_report(fit, figures=False)
    for term in fit.term_names:
        assert f"<td>{term}</td>" in html
    assert not _embedded_pngs(html)
    # one row per run in the run sheet (rows numbered 1..n)
    assert f"<td>{fit.design.n_runs}</td>" in html


def test_report_multiple_responses_share_one_design_section(ccd_fits):
    html = analysis_report(list(ccd_fits))
    assert 'id="yield-pct"' in html and 'id="impurity"' in html
    assert html.count("<h2>Design</h2>") == 1


def test_report_mapping_headings_and_extras(ccd_fits, tmp_path):
    fit, _ = ccd_fits
    fig = Figure()
    fig.add_subplot().plot([0, 1], [0, 1])
    out = tmp_path / "report.html"
    html = analysis_report(
        {"Yield model": fit},
        out,
        title="My study",
        description="First paragraph.\n\nSecond <para>.",
        extras={
            "Optimum": fit.optimum(),
            "Custom figure": fig,
            "Table": pd.DataFrame({"a": [1.0]}),
            "Notes": "Ran on Tuesday.",
        },
    )
    assert out.read_text(encoding="utf-8") == html
    assert "<title>My study</title>" in html
    assert "Second &lt;para&gt;." in html  # text is escaped
    for heading in ["Yield model", "Optimum", "Custom figure", "Table", "Notes"]:
        assert f"<h2>{heading}</h2>" in html
    assert "Optimum(max:" in html  # repr shown under the to_frame table


def test_report_svg_format(ccd_fits):
    fit, _ = ccd_fits
    html = analysis_report(fit, image_format="svg")
    assert "<svg" in html and "<?xml" not in html
    assert not _embedded_pngs(html)
    with pytest.raises(ValueError):
        analysis_report(fit, image_format="jpg")  # type: ignore[arg-type]


def test_report_saturated_fit_notes_missing_inference():
    design = full_factorial([ContinuousFactor("A", 0, 1), ContinuousFactor("B", 0, 1)])
    y = np.array([1.0, 3.0, 2.0, 7.0])
    with pytest.warns(UserWarning):
        fit = fit_ols(design, y)
    html = analysis_report(fit)
    assert "Saturated model" in html
    assert "Residuals vs fitted" not in html  # no residual diagnostics without error df
    assert "Pareto" in html


def test_report_mixture_fit_draws_ternary_and_drops_effects():
    design = simplex_lattice([MixtureFactor(n) for n in "ABC"], degree=2)
    c = design.runs
    y = (5 * c.A + 3 * c.B + 4 * c.C + 6 * c.A * c.B).to_numpy()
    y = y + np.random.default_rng(1).normal(0, 0.05, len(y))
    fit = fit_ols(design.replicate(2), np.tile(y, 2), model="scheffe-quadratic")
    html = analysis_report(fit)
    assert "Scheffé quadratic blending" in html
    assert "Ternary contour" in html
    assert "<th>effect</th>" not in html  # effects are meaningless on proportions


def test_fit_report_method_matches_function(ccd_fits):
    fit, _ = ccd_fits
    a = fit.report(figures=False)
    b = analysis_report(fit, figures=False)
    strip = lambda s: re.sub(r"generated [0-9T:-]+", "", s)  # noqa: E731
    assert strip(a) == strip(b)


def test_report_rejects_empty_and_wrong_types():
    with pytest.raises(ValueError):
        analysis_report([])
    with pytest.raises(TypeError):
        analysis_report(["not a fit"])  # type: ignore[list-item]
