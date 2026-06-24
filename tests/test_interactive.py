"""Tests for the Tier-1 interactive HTML output."""

from __future__ import annotations

from doe import ContinuousFactor, central_composite, full_factorial
from doe.interactive import to_html


def _two_factor_design():
    return full_factorial(
        [
            ContinuousFactor("temp", 20.0, 60.0, units="C"),
            ContinuousFactor("time", 5.0, 15.0),
        ]
    )


def test_to_html_is_self_contained_and_lists_every_run():
    design = _two_factor_design()
    html = to_html(design)

    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # one <tr> in <tbody> per run (header rows live in <thead>)
    assert html.count("<tr>") >= design.n_runs
    # natural-unit header carries the factor units
    assert "temp (C)" in html
    # coded columns present by default
    assert "temp [coded]" in html
    # DataTables wiring is included
    assert "DataTable(" in html
    assert "#T_doe-design" in html


def test_to_html_writes_file(tmp_path):
    design = _two_factor_design()
    out = tmp_path / "design.html"
    returned = to_html(design, path=out)

    assert out.exists()
    assert out.read_text(encoding="utf-8") == returned


def test_to_html_omits_coded_and_cdn_when_disabled():
    design = _two_factor_design()
    html = to_html(design, coded=False, cdn=False)

    assert "[coded]" not in html
    assert "datatables" not in html.lower()
    assert "DataTable(" not in html


def test_to_html_tints_point_types_for_rsm_designs():
    design = central_composite(
        [ContinuousFactor("x1", -1.0, 1.0), ContinuousFactor("x2", -1.0, 1.0)]
    )
    assert design.point_types is not None  # CCD tags runs

    html = to_html(design)
    assert "type" in html
    # the center-point tint colour appears in the styled cells
    assert "#fff3cd" in html


def test_to_html_surfaces_std_order_for_randomised_designs():
    design = _two_factor_design()
    assert "std_order" not in to_html(design)  # plain design has no std_order column

    plate_order = design.randomize(seed=0)
    html = to_html(plate_order)
    assert "<th>std_order</th>" in html
    # every original design-row index is rendered back into the table
    for std in plate_order.runs["std_order"]:
        assert f">{std}</td>" in html
