"""Smoke tests for the flat public API exported from ``doe``."""

import doe


def test_phase3_public_api_exports_are_available():
    for name in [
        "Efficiency",
        "OptimalDesign",
        "augment",
        "candidate_grid",
        "condition_number",
        "coordinate_exchange",
        "correlation_matrix",
        "d_optimal",
        "efficiency",
        "i_optimal",
        "information_matrix",
        "leverage",
        "log_det_information",
        "vif",
    ]:
        assert name in doe.__all__
        assert getattr(doe, name) is not None


def test_phase5_public_api_exports_are_available():
    for name in [
        "definitive_screening",
        "split_plot",
        "fit_gls",
        "randomized_complete_block",
        "latin_square",
        "blocked_factorial",
    ]:
        assert name in doe.__all__
        assert getattr(doe, name) is not None
