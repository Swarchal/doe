"""Self-contained interactive HTML output of a :class:`~doe.design.Design`.

Tier 1: a styled, sortable/searchable table of the runs in both natural and coded
units. Pure ``pandas``/``numpy``/stdlib -- no extra Python dependencies (the table is
emitted by hand rather than through ``DataFrame.style``, which would require ``jinja2``,
and the gradient colouring is hand-rolled rather than going through matplotlib colormaps).
Interactivity (sort / filter / search) is provided by the DataTables JS library, loaded
from a CDN by default; pass ``cdn=False`` for a static styled table with no external
assets (e.g. offline/air-gapped use).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from .design import Design
from .factors import ContinuousFactor

# coolwarm-ish diverging endpoints for coded values in [-1, +1]
_LOW = (59, 76, 192)  # blue  -> -1
_MID = (247, 247, 247)  # white ->  0
_HIGH = (180, 4, 38)  # red   -> +1

# soft tints for run/point types
_TYPE_COLORS = {
    "center": "#fff3cd",
    "axial": "#d1ecf1",
    "corner": "#e2e3e5",
    "factorial": "#e2e3e5",
}

_DATATABLES_CSS = "https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"
_JQUERY_JS = "https://code.jquery.com/jquery-3.7.1.min.js"
_DATATABLES_JS = "https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _diverging_hex(value: float, vmax: float) -> str:
    """Map a coded ``value`` to a blue-white-red hex colour, normalised by ``vmax``."""
    if vmax <= 0 or not np.isfinite(value):
        return ""
    t = max(-1.0, min(1.0, value / vmax))
    r, g, b = _lerp(_MID, _HIGH, t) if t >= 0 else _lerp(_MID, _LOW, -t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(value: object) -> str:
    """Format a cell value -- 4 significant figures for floats, str otherwise."""
    if isinstance(value, float | np.floating):
        return f"{float(value):.4g}"
    return str(value)


def _build_table_html(design: Design, *, coded: bool) -> str:
    """Emit the ``<table>`` markup with inline gradient/type colouring."""
    runs = design.runs.reset_index(drop=True)
    coded_runs = design.coded().reset_index(drop=True)
    point_types = design.point_types

    # header labels paired with the data they pull from
    headers: list[str] = ["run"]
    columns: list[tuple[str, np.ndarray]] = []  # (kind, values); kind in {plain, natural, coded}

    # a randomised design carries the original design-row index so readouts can be
    # re-joined after pipetting in shuffled order -- surface it next to the run number.
    if "std_order" in runs.columns:
        headers.append("std_order")
        columns.append(("plain", runs["std_order"].to_numpy()))

    for factor in design.factors:
        label = factor.name
        if isinstance(factor, ContinuousFactor) and factor.units:
            label = f"{factor.name} ({factor.units})"
        headers.append(label)
        columns.append(("natural", runs[factor.name].to_numpy()))
        if coded:
            headers.append(f"{factor.name} [coded]")
            columns.append(("coded", coded_runs[factor.name].to_numpy()))
    if point_types is not None:
        headers.append("type")

    coded_values = [v for kind, v in columns if kind == "coded"]
    vmax = 1.0
    if coded_values:
        stacked = np.abs(np.concatenate([np.asarray(v, dtype=float) for v in coded_values]))
        vmax = float(np.nanmax(stacked)) or 1.0

    head = "".join(f"<th>{escape(h)}</th>" for h in headers)

    body_rows: list[str] = []
    for i in range(design.n_runs):
        cells = [f"<td>{i + 1}</td>"]
        for kind, values in columns:
            value = values[i]
            style = ""
            if kind == "coded":
                colour = _diverging_hex(float(value), vmax)
                if colour:
                    style = f' style="background-color:{colour}"'
            cells.append(f"<td{style}>{escape(_fmt(value))}</td>")
        if point_types is not None:
            tint = _TYPE_COLORS.get(str(point_types[i]), "")
            style = f' style="background-color:{tint}"' if tint else ""
            cells.append(f"<td{style}>{escape(str(point_types[i]))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<table id="T_doe-design" class="display compact" style="width:100%">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def to_html(
    design: Design,
    path: str | Path | None = None,
    *,
    coded: bool = True,
    title: str | None = None,
    cdn: bool = True,
) -> str:
    """Render ``design`` as a self-contained interactive HTML document.

    The table shows one row per run with natural-unit factor columns (and coded-unit
    columns when ``coded=True``), coloured by a diverging scale so the design structure
    is visible at a glance, and tinted by point type when the design tracks them. The
    output is sortable, searchable and pageable via DataTables.

    Parameters
    ----------
    design : Design
        The design to render.
    path : str | Path | None
        If given, the HTML is also written to this file. The string is always returned.
    coded : bool
        Include coded-unit columns (default ``True``).
    title : str | None
        Document heading; defaults to the design's ``name`` or ``"Experimental design"``.
    cdn : bool
        Load DataTables + jQuery from a CDN (default). Set ``False`` to omit the
        ``<script>``/``<link>`` tags entirely (e.g. for offline/air-gapped use), which
        yields a static styled table.
    """
    table_html = _build_table_html(design, coded=coded)
    heading = escape(title or design.name or "Experimental design")

    head_assets = ""
    init_script = ""
    if cdn:
        head_assets = (
            f'<link rel="stylesheet" href="{_DATATABLES_CSS}">\n'
            f'<script src="{_JQUERY_JS}"></script>\n'
            f'<script src="{_DATATABLES_JS}"></script>'
        )
        init_script = (
            "<script>$(function(){"
            '$("#T_doe-design").DataTable({"paging":true,"pageLength":25,"order":[]});'
            "});</script>"
        )

    return _write_document(
        document=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{heading}</title>
{head_assets}
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
h1 {{ font-size: 1.4rem; }}
table.display td, table.display th {{ padding: 4px 10px; text-align: right; }}
</style>
</head>
<body>
<h1>{heading}</h1>
<p>{design.n_runs} runs &middot; {len(design.factors)} factors</p>
{table_html}
{init_script}
</body>
</html>
""",
        path=path,
    )


def _write_document(document: str, path: str | Path | None) -> str:
    if path is not None:
        Path(path).write_text(document, encoding="utf-8")
    return document
