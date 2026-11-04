"""Self-contained HTML reports of analysed results, with embedded figures.

:func:`analysis_report` turns one or more fitted models (:class:`~doe.analysis.fit.FitResult`)
into a single HTML file you can email, archive or open offline: fit statistics, the
coefficient table (with confidence intervals), the ANOVA and lack-of-fit tests, the standard
diagnostic figures, and the design's run sheet with its measured responses. Figures are
rendered with matplotlib and embedded inline (base64 PNG or inline SVG), so the file has no
external assets at all.

Extra material -- an :class:`~doe.analysis.optimize.Optimum`, a desirability result, a
DataFrame, a matplotlib figure, or a paragraph of notes -- can be appended via ``extras``.

Figures need the optional ``matplotlib`` dependency (``pip install 'doe[plotting]'``); pass
``figures=False`` for a tables-only report without it.
"""

from __future__ import annotations

import base64
import io
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .analysis.fit import FitResult
from .design import Design
from .factors import CategoricalFactor, ContinuousFactor, MixtureFactor

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ImageFormat = Literal["png", "svg"]

#: p-values below this are highlighted in the coefficient and ANOVA tables.
SIGNIFICANCE = 0.05

_CSS = """
:root {
  --bg: #f7f7f5; --surface: #ffffff; --text: #1d1d1b; --muted: #66645e;
  --border: #e2e0da; --accent: #2b59c3; --sig-bg: #e6eefc; --sig-text: #173a8a;
  --warn-bg: #fff4e0; --warn-text: #7a4b00;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --surface: #20201e; --text: #ecebe6; --muted: #a3a19a;
    --border: #36352f; --accent: #8aa9f0; --sig-bg: #1f2c4a; --sig-text: #c8d7fb;
    --warn-bg: #3a2c10; --warn-text: #f3d08f;
  }
}
:root[data-theme="dark"] {
  --bg: #161615; --surface: #20201e; --text: #ecebe6; --muted: #a3a19a;
  --border: #36352f; --accent: #8aa9f0; --sig-bg: #1f2c4a; --sig-text: #c8d7fb;
  --warn-bg: #3a2c10; --warn-text: #f3d08f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 2rem 16px 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.25rem; margin: 2.5rem 0 .75rem; padding-bottom: .3rem;
  border-bottom: 1px solid var(--border); }
h3 { font-size: 1rem; margin: 1.5rem 0 .5rem; color: var(--muted); font-weight: 600; }
.meta { color: var(--muted); font-size: .85rem; margin: 0; }
nav { margin: 1rem 0 0; font-size: .9rem; }
nav a { color: var(--accent); margin-right: 1rem; text-decoration: none; }
.stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: .5rem; margin: .5rem 0 1rem; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  padding: .5rem .75rem; }
.stat .label { color: var(--muted); font-size: .75rem; text-transform: uppercase;
  letter-spacing: .03em; }
.stat .value { font-size: 1.15rem; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: .85rem;
  font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--border); }
th { position: sticky; top: 0; background: var(--surface); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tr:last-child td { border-bottom: none; }
tr.sig td { background: var(--sig-bg); color: var(--sig-text); }
.figures { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1rem; }
figure { margin: 0; background: #ffffff; border: 1px solid var(--border); border-radius: 6px;
  padding: .5rem; }
figure img, figure svg { display: block; width: 100%; height: auto; }
.note { background: var(--warn-bg); color: var(--warn-text); border-radius: 6px;
  padding: .5rem .75rem; font-size: .85rem; margin: .5rem 0; }
p.text { max-width: 70ch; }
pre { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  padding: .75rem; overflow-x: auto; font-size: .85rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem; }
"""


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #


def _fmt(value: object) -> str:
    """4 significant figures for floats, blank for NaN, ``str`` otherwise."""
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, float | np.floating):
        v = float(value)
        if np.isnan(v):
            return "—"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(value)


def _frame_table(frame: pd.DataFrame, *, p_column: str | None = None, index: bool = True) -> str:
    """Render a DataFrame as an HTML table; rows with ``p_column < SIGNIFICANCE`` are tinted."""
    cols = [str(c) for c in frame.columns]
    index_name = str(frame.index.name) if frame.index.name is not None else ""
    head = (f"<th>{escape(index_name)}</th>" if index else "") + "".join(
        f"<th>{escape(c)}</th>" for c in cols
    )
    rows: list[str] = []
    for label, row in frame.iterrows():
        cls = ""
        if p_column is not None and p_column in frame.columns:
            p = row[p_column]
            if isinstance(p, float | np.floating) and np.isfinite(p) and p < SIGNIFICANCE:
                cls = ' class="sig"'
        cells = (f"<td>{escape(_fmt(label))}</td>" if index else "") + "".join(
            f"<td>{escape(_fmt(v))}</td>" for v in row.to_numpy()
        )
        rows.append(f"<tr{cls}>{cells}</tr>")
    return (
        '<div class="scroll"><table>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _stat_cards(stats: Sequence[tuple[str, object]]) -> str:
    cards = "".join(
        f'<div class="stat"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(_fmt(value))}</div></div>'
        for label, value in stats
    )
    return f'<div class="stats">{cards}</div>'


def _note(text: str) -> str:
    return f'<p class="note">{escape(text)}</p>'


def _slug(text: str, used: set[str]) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-") or "section"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


# --------------------------------------------------------------------------- #
# figure embedding
# --------------------------------------------------------------------------- #


def _new_figure(figsize: tuple[float, float] = (5.5, 4.0)) -> Figure:
    """A pyplot-free Figure: no GUI backend is touched and nothing leaks into pyplot state."""
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "report figures need matplotlib (pip install 'doe[plotting]'); "
            "pass figures=False for a tables-only report"
        ) from exc
    return Figure(figsize=figsize, layout="constrained")


def _embed_figure(fig: Figure, *, image_format: ImageFormat, dpi: int, caption: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=image_format, dpi=dpi, facecolor="white")
    alt = escape(caption, quote=True)
    if image_format == "svg":
        svg = buf.getvalue().decode("utf-8")
        body = svg[svg.find("<svg") :]
        return f'<figure role="img" aria-label="{alt}">{body}</figure>'
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<figure><img alt="{alt}" src="data:image/png;base64,{data}"></figure>'


def _figure_of(obj: object) -> Figure | None:
    """The matplotlib Figure behind ``obj`` (a Figure, SubFigure or Axes), else ``None``."""
    try:
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure, SubFigure
    except ImportError:  # pragma: no cover
        return None
    if isinstance(obj, Figure):
        return obj
    if isinstance(obj, SubFigure):
        return obj.figure
    if isinstance(obj, Axes):
        return obj.figure  # type: ignore[return-value]
    return None


# --------------------------------------------------------------------------- #
# per-fit content
# --------------------------------------------------------------------------- #


def _model_label(fit: FitResult) -> str:
    if fit.term_names and fit.term_names[0] != "Intercept":
        return "Scheffé quadratic blending" if fit.order >= 2 else "Scheffé linear blending"
    parts = ["quadratic" if fit.order >= 2 else "linear"]
    if fit.interactions:
        parts.append("two-factor interactions")
    label = " + ".join(parts)
    return f"{label} (GLS split-plot)" if fit.dof_terms is not None else label


def _try(func: Callable[[], Any]) -> Any:
    """Run an optional statistic; ``None`` when it does not apply to this fit."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func()
    except (ValueError, NotImplementedError, np.linalg.LinAlgError, ZeroDivisionError):
        return None


def _top_continuous_pair(fit: FitResult) -> tuple[str, str] | None:
    """The two continuous factors with the largest main-effect coefficients."""
    continuous = [f.name for f in fit.factors if type(f) is ContinuousFactor]
    if len(continuous) < 2:
        return None
    coef = dict(zip(fit.term_names, np.abs(fit.coefficients), strict=True))
    ranked = sorted(continuous, key=lambda n: -float(coef.get(n, 0.0)))
    return ranked[0], ranked[1]


def _auto_figures(fit: FitResult) -> list[tuple[str, Callable[[Axes], object]]]:
    """Pick the diagnostic figures that make sense for ``fit``."""
    from . import plotting

    specs: list[tuple[str, Callable[[Axes], object]]] = []
    n_terms = len(fit.term_names) - (1 if fit.term_names[0] == "Intercept" else 0)
    effects_ok = n_terms > 0 and bool(np.isfinite(fit.effects[1:]).any())
    if effects_ok and fit.term_names[0] == "Intercept":
        specs.append(("Pareto plot of effects", lambda ax: plotting.pareto_plot(fit, ax=ax)))
        specs.append(("Half-normal plot", lambda ax: plotting.half_normal_plot(fit, ax=ax)))
    if fit.dof_resid > 0:
        specs.append(("Predicted vs actual", lambda ax: plotting.predicted_vs_actual(fit, ax=ax)))
        specs.append(("Residuals vs fitted", lambda ax: plotting.residuals_vs_fitted(fit, ax=ax)))
        specs.append(("Normal Q-Q of residuals", lambda ax: plotting.normal_qq(fit, ax=ax)))
    specs.append(("Leverage by run", lambda ax: plotting.leverage_plot(fit, ax=ax)))

    if fit.factors.is_mixture:
        if len(fit.factors) == 3:
            specs.append(
                (
                    "Ternary contour of the fitted blend",
                    lambda ax: plotting.ternary_contour(fit, fit.design, ax=ax),
                )
            )
    elif fit.order >= 2 or fit.interactions:
        pair = _top_continuous_pair(fit)
        if pair is not None:
            x, y = pair
            specs.append(
                (
                    f"Fitted surface: {x} × {y}",
                    lambda ax: plotting.contour_plot(fit, x, y, ax=ax),
                )
            )
            if fit.order < 2:
                specs.append(
                    (
                        f"Interaction: {x} × {y}",
                        lambda ax: plotting.interaction_plot(fit, x, y, ax=ax),
                    )
                )
    return specs


def _fit_section(
    fit: FitResult,
    heading: str,
    anchor: str,
    *,
    figures: bool,
    image_format: ImageFormat,
    dpi: int,
    confidence: float,
) -> str:
    parts = [f'<section id="{anchor}"><h2>{escape(heading)}</h2>']
    n_runs = len(fit.fitted)
    parts.append(
        f'<p class="meta">Model: {escape(_model_label(fit))} &middot; {n_runs} runs &middot; '
        f"{len(fit.term_names)} terms &middot; {fit.dof_resid} residual df</p>"
    )

    lof = _try(fit.lack_of_fit)
    stats: list[tuple[str, object]] = [
        ("R²", fit.r_squared),
        ("Adjusted R²", _try(fit.adjusted_r2)),
        ("Predicted R²", _try(fit.predicted_r2)),
        ("RMSE", float(np.sqrt(fit.mse)) if np.isfinite(fit.mse) else float("nan")),
        ("PRESS", _try(fit.press)),
    ]
    if fit.sigma2_wp is not None:
        stats.append(("Whole-plot σ²", fit.sigma2_wp))
        stats.append(("Sub-plot σ²", fit.mse))
    if lof is not None:
        stats.append(("Lack-of-fit p", lof.p_value))
    parts.append(_stat_cards([(k, v) for k, v in stats if v is not None]))

    if fit.dof_resid == 0:
        parts.append(
            _note(
                "Saturated model (0 residual df): no error estimate, so standard errors, "
                "p-values and residual diagnostics are unavailable. Judge effects from the "
                "Pareto / half-normal plots."
            )
        )

    # coefficients + confidence intervals
    pct = f"{confidence:.0%}"
    table = fit.summary()
    ci = fit.conf_int(confidence).rename(
        columns={"lower": f"{pct} CI lower", "upper": f"{pct} CI upper"}
    )
    table = table.join(ci)
    if not np.isfinite(table["effect"]).any():
        table = table.drop(columns="effect")
    parts.append("<h3>Coefficients (coded units)</h3>")
    parts.append(_frame_table(table, p_column="p"))
    parts.append(
        f'<p class="meta">Highlighted rows have p &lt; {SIGNIFICANCE}. In coded units an effect '
        "is 2 × the coefficient (the −1 → +1 swing).</p>"
    )

    anova = _try(fit.anova)
    if anova is not None:
        parts.append("<h3>ANOVA (sequential, Type I)</h3>")
        parts.append(_frame_table(anova, p_column="p"))
    if lof is not None:
        lof_frame = pd.DataFrame(
            {
                "SS": [lof.ss_lof, lof.ss_pe],
                "df": [lof.df_lof, lof.df_pe],
                "F": [lof.f_stat, float("nan")],
                "p": [lof.p_value, float("nan")],
            },
            index=pd.Index(["Lack of fit", "Pure error"], name="source"),
        )
        parts.append("<h3>Lack of fit</h3>")
        parts.append(_frame_table(lof_frame))
        verdict = "adequate" if lof.p_value >= SIGNIFICANCE else "showing significant lack of fit"
        parts.append(f'<p class="meta">The model looks {verdict} (p = {_fmt(lof.p_value)}).</p>')

    if figures:
        rendered: list[str] = []
        skipped: list[str] = []
        for caption, draw in _auto_figures(fit):
            fig = _new_figure()
            ax = fig.add_subplot()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    draw(ax)
            except (ValueError, NotImplementedError, TypeError, np.linalg.LinAlgError):
                skipped.append(caption)
                continue
            rendered.append(_embed_figure(fig, image_format=image_format, dpi=dpi, caption=caption))
        if rendered:
            parts.append("<h3>Figures</h3>")
            parts.append(f'<div class="figures">{"".join(rendered)}</div>')
        if skipped:
            parts.append(_note("Not applicable to this fit: " + ", ".join(skipped) + "."))

    parts.append("</section>")
    return "".join(parts)


def _design_section(design: Design, heading: str, anchor: str) -> str:
    runs = design.runs.reset_index(drop=True).copy()
    if design.point_types is not None:
        runs["type"] = list(design.point_types)
    if design.whole_plots is not None:
        runs["whole_plot"] = list(design.whole_plots)
    runs.index = pd.RangeIndex(1, len(runs) + 1, name="run")
    meta = [f"{design.n_runs} runs", f"{len(design.factors)} factors"]
    if design.meta.get("randomized"):
        meta.append(f"randomized (seed {design.meta.get('random_seed')})")
    factor_rows = []
    for f in design.factors:
        if isinstance(f, CategoricalFactor):
            factor_rows.append((f.name, "categorical", ", ".join(map(str, f.levels))))
        elif isinstance(f, MixtureFactor):
            factor_rows.append((f.name, "mixture", f"{_fmt(f.low)} – {_fmt(f.high)}"))
        elif isinstance(f, ContinuousFactor):
            units = f" {f.units}" if f.units else ""
            factor_rows.append((f.name, "continuous", f"{_fmt(f.low)} – {_fmt(f.high)}{units}"))
    factors = pd.DataFrame(factor_rows, columns=pd.Index(["factor", "type", "range / levels"]))
    return (
        f'<section id="{anchor}"><h2>{escape(heading)}</h2>'
        f'<p class="meta">{escape(" · ".join(meta))}</p>'
        "<h3>Factors</h3>"
        f"{_frame_table(factors, index=False)}"
        "<h3>Runs (natural units, with responses)</h3>"
        f"{_frame_table(runs)}</section>"
    )


def _extra_body(heading: str, obj: object, *, image_format: ImageFormat, dpi: int) -> str:
    if isinstance(obj, list):
        return "".join(
            _extra_body(heading, part, image_format=image_format, dpi=dpi) for part in obj
        )
    fig = _figure_of(obj)
    if fig is not None:
        return (
            '<div class="figures">'
            f"{_embed_figure(fig, image_format=image_format, dpi=dpi, caption=heading)}</div>"
        )
    if isinstance(obj, pd.DataFrame):
        plain_index = isinstance(obj.index, pd.RangeIndex) and obj.index.name is None
        return _frame_table(
            obj, p_column="p" if "p" in obj.columns else None, index=not plain_index
        )
    if isinstance(obj, pd.Series):
        return _frame_table(obj.to_frame())
    if isinstance(obj, str):
        return "".join(f'<p class="text">{escape(p)}</p>' for p in obj.split("\n\n") if p.strip())
    if callable(getattr(obj, "to_frame", None)):
        frame = obj.to_frame()  # type: ignore[attr-defined]
        return _frame_table(frame, index=False) + f"<pre>{escape(repr(obj))}</pre>"
    return f"<pre>{escape(repr(obj))}</pre>"


def _extra_section(
    heading: str, obj: object, anchor: str, *, image_format: ImageFormat, dpi: int
) -> str:
    body = _extra_body(heading, obj, image_format=image_format, dpi=dpi)
    return f'<section id="{anchor}"><h2>{escape(heading)}</h2>{body}</section>'


def _normalize_results(
    results: FitResult | Sequence[FitResult] | Mapping[str, FitResult],
) -> list[tuple[str, FitResult]]:
    if isinstance(results, FitResult):
        items: list[tuple[str | None, FitResult]] = [(None, results)]
    elif isinstance(results, Mapping):
        items = [(str(k), v) for k, v in results.items()]
    else:
        items = [(None, r) for r in results]
    if not items:
        raise ValueError("analysis_report needs at least one FitResult")
    out: list[tuple[str, FitResult]] = []
    for i, (label, fit) in enumerate(items, start=1):
        if not isinstance(fit, FitResult):
            raise TypeError(f"expected FitResult, got {type(fit).__name__}")
        name = label or fit.response_name or ("Response" if len(items) == 1 else f"Response {i}")
        out.append((name, fit))
    return out


def _normalize_extras(
    extras: Mapping[str, object] | Sequence[object] | None,
) -> list[tuple[str, object]]:
    if extras is None:
        return []
    if isinstance(extras, Mapping):
        return [(str(k), v) for k, v in extras.items()]
    out: list[tuple[str, object]] = []
    for obj in extras:
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
            out.append((obj[0], obj[1]))
        elif isinstance(obj, str):
            out.append(("Notes", obj))
        else:
            name = type(obj).__name__
            out.append(("Figure" if _figure_of(obj) is not None else name, obj))
    return out


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #


def analysis_report(
    results: FitResult | Sequence[FitResult] | Mapping[str, FitResult],
    path: str | Path | None = None,
    *,
    title: str | None = None,
    description: str | None = None,
    extras: Mapping[str, object] | Sequence[object] | None = None,
    figures: bool = True,
    include_design: bool = True,
    image_format: ImageFormat = "png",
    dpi: int = 110,
    confidence: float = 0.95,
) -> str:
    """Render fitted results as one self-contained HTML report with embedded figures.

    For each fit the report shows headline statistics (R², adjusted/predicted R², RMSE, PRESS,
    lack-of-fit p), the coefficient table with confidence intervals (significant rows
    highlighted), the sequential ANOVA and lack-of-fit tables where they apply, and the
    diagnostic figures that suit the model: Pareto + half-normal effect plots, predicted vs
    actual, residuals vs fitted, normal Q-Q, leverage, and a contour (or ternary contour for a
    3-component mixture) over the two most influential continuous factors. Statistics or
    figures that do not apply (a saturated model, a split-plot GLS fit, a categorical axis) are
    skipped with a note rather than failing the report. The originating design is listed once
    per distinct design, with its factors and the measured responses.

    Args:
        results: A :class:`~doe.analysis.fit.FitResult`, a sequence of them (e.g. one per
            response), or a ``{heading: fit}`` mapping. Headings default to each fit's
            ``response_name``.
        path: If given, the HTML is also written to this file. The string is always returned.
        title: Report title; defaults to the design's ``name`` or ``"DoE analysis report"``.
        description: Optional introductory paragraph(s) shown under the title (blank lines
            separate paragraphs).
        extras: Additional sections appended after the fits, as a ``{heading: obj}`` mapping or
            a sequence of objects / ``(heading, obj)`` pairs. Supported objects: matplotlib
            ``Figure``/``Axes`` (embedded as an image), ``DataFrame``/``Series`` (table),
            anything with a ``to_frame()`` method such as
            :class:`~doe.analysis.optimize.Optimum` or
            :class:`~doe.analysis.optimize.DesirabilityResult` (table + repr), ``str`` (notes),
            a ``list`` of any of these (rendered in order under the one heading), and
            anything else (its ``repr``).
        figures: Include the automatic diagnostic figures (needs matplotlib). ``extras``
            figures are always embedded.
        include_design: Include the design's factor list and run sheet.
        image_format: ``"png"`` (base64, default; compact for busy plots) or ``"svg"``
            (inline vector graphics; crisp at any zoom).
        dpi: Resolution of PNG figures.
        confidence: Level for the coefficient confidence intervals.

    Returns:
        The complete HTML document.

    Examples:
        >>> from doe import ContinuousFactor, central_composite, fit_ols
        >>> from doe.report import analysis_report
        >>> design = central_composite(
        ...     [ContinuousFactor("temp", 40, 80), ContinuousFactor("time", 5, 15)], seed=1
        ... )  # doctest: +SKIP
        >>> fit = fit_ols(design.with_response("yield", y), "yield", order=2)  # doctest: +SKIP
        >>> html = analysis_report(fit, "report.html", extras=[fit.optimum()])  # doctest: +SKIP
    """
    if image_format not in ("png", "svg"):
        raise ValueError("image_format must be 'png' or 'svg'")
    fits = _normalize_results(results)
    extra_items = _normalize_extras(extras)

    first_design = fits[0][1].design
    heading = title or (first_design.name if first_design and first_design.name else None)
    heading = heading or "DoE analysis report"

    used: set[str] = set()
    toc: list[tuple[str, str]] = []
    sections: list[str] = []

    for name, fit in fits:
        anchor = _slug(name, used)
        toc.append((anchor, name))
        sections.append(
            _fit_section(
                fit,
                name,
                anchor,
                figures=figures,
                image_format=image_format,
                dpi=dpi,
                confidence=confidence,
            )
        )

    for name, obj in extra_items:
        anchor = _slug(name, used)
        toc.append((anchor, name))
        sections.append(_extra_section(name, obj, anchor, image_format=image_format, dpi=dpi))

    if include_design:
        designs: list[Design] = []
        for _, fit in fits:
            if fit.design is not None and all(fit.design is not d for d in designs):
                designs.append(fit.design)
        for i, design in enumerate(designs, start=1):
            name = "Design" if len(designs) == 1 else f"Design {i}"
            anchor = _slug(name, used)
            toc.append((anchor, name))
            sections.append(_design_section(design, name, anchor))

    from . import __version__

    intro = ""
    if description:
        intro = "".join(
            f'<p class="text">{escape(p)}</p>' for p in description.split("\n\n") if p.strip()
        )
    nav = "".join(f'<a href="#{a}">{escape(n)}</a>' for a, n in toc)
    stamp = datetime.now().isoformat(timespec="seconds")
    responses = ", ".join(name for name, _ in fits)
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(heading)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>{escape(heading)}</h1>
<p class="meta">Responses: {escape(responses)} &middot; generated {stamp}</p>
{intro}
<nav>{nav}</nav>
{"".join(sections)}
<footer>Generated by doe {escape(__version__)}.</footer>
</main>
</body>
</html>
"""
    if path is not None:
        Path(path).write_text(document, encoding="utf-8")
    return document
