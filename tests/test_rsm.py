"""Phase 2a: response-surface design generators (CCD + Box-Behnken).

Written test-first to document the intended API and the expected run structures.
Correctness anchors follow Montgomery's RSM chapters (run counts, axial distances).
"""

import numpy as np
import pytest

from doe.factors import ContinuousFactor
from doe.generators.rsm import box_behnken, central_composite


def _factors(n, low=0.0, high=10.0):
    return [ContinuousFactor(chr(ord("a") + i), low=low, high=high) for i in range(n)]


# --------------------------------------------------------------------------- #
# Central composite design -- face-centered default (alpha = 1)
# --------------------------------------------------------------------------- #


def test_ccd_faced_run_count_k2():
    # factorial 2^2 = 4, axial 2k = 4, center = 4  ->  12
    design = central_composite(_factors(2))
    assert design.n_runs == 12
    assert design.factors.names == ["a", "b"]


def test_ccd_faced_run_count_k3():
    # factorial 2^3 = 8, axial 2k = 6, center = 4  ->  18
    design = central_composite(_factors(3))
    assert design.n_runs == 18


def test_ccd_faced_coded_levels_are_minus1_0_1():
    design = central_composite(_factors(2))
    coded = design.coded().to_numpy()
    assert set(np.unique(coded)).issubset({-1.0, 0.0, 1.0})
    # a face-centered CCD genuinely uses all three coded levels
    assert set(np.unique(coded)) == {-1.0, 0.0, 1.0}


def test_ccd_faced_stays_inside_bounds():
    design = central_composite(_factors(2, low=0.0, high=10.0))
    runs = design.runs[design.factors.names].to_numpy()
    assert runs.min() >= 0.0
    assert runs.max() <= 10.0
    # axial points sit exactly on the stated bounds (no extrapolation)
    assert np.isclose(runs.min(), 0.0)
    assert np.isclose(runs.max(), 10.0)


def test_ccd_faced_meta_marks_no_extrapolation():
    design = central_composite(_factors(2))
    assert np.isclose(design.meta["alpha"], 1.0)
    assert design.meta["axial_extrapolates"] is False


def test_ccd_center_count_configurable():
    design = central_composite(_factors(2), center=6)
    assert design.n_runs == 4 + 4 + 6
    assert design.n_center == 6
    assert len(design.center_indices) == 6


def test_ccd_center_points_decode_to_factor_centers():
    design = central_composite(_factors(2, low=0.0, high=10.0), center=4)
    centers = design.runs.iloc[design.center_indices][design.factors.names].to_numpy()
    assert np.allclose(centers, 5.0)


# --------------------------------------------------------------------------- #
# Central composite design -- rotatable / orthogonal (opt-in, extrapolating)
# --------------------------------------------------------------------------- #


def test_ccd_rotatable_alpha_k2():
    design = central_composite(_factors(2), alpha="rotatable")
    coded = design.coded().to_numpy()
    expected_alpha = 4.0**0.25  # (n_factorial)**0.25 = sqrt(2)
    assert np.isclose(np.abs(coded).max(), expected_alpha)
    assert np.isclose(design.meta["alpha"], expected_alpha)


def test_ccd_rotatable_alpha_k3():
    design = central_composite(_factors(3), alpha="rotatable")
    coded = design.coded().to_numpy()
    assert np.isclose(np.abs(coded).max(), 8.0**0.25)  # ~1.6818


def test_ccd_rotatable_extrapolates_beyond_bounds():
    design = central_composite(_factors(2, low=0.0, high=10.0), alpha="rotatable")
    runs = design.runs[design.factors.names].to_numpy()
    # axial points fall outside the stated [0, 10] box (circumscribed CCD)
    assert runs.max() > 10.0
    assert runs.min() < 0.0
    assert design.meta["axial_extrapolates"] is True


def test_ccd_explicit_float_alpha():
    design = central_composite(_factors(2), alpha=1.5)
    coded = design.coded().to_numpy()
    assert np.isclose(np.abs(coded).max(), 1.5)


# --------------------------------------------------------------------------- #
# Central composite design -- fractional core
# --------------------------------------------------------------------------- #


def test_ccd_fractional_core():
    # 5 factors, resolution-V half fraction core: 2^(5-1) = 16, axial 10, center 4 -> 30
    design = central_composite(_factors(5), fraction=["E=ABCD"], center=4)
    assert design.n_runs == 16 + 10 + 4


# --------------------------------------------------------------------------- #
# Box-Behnken design
# --------------------------------------------------------------------------- #


def test_box_behnken_k3_run_count():
    # 3 factor-pairs x 4 edge runs = 12, plus 3 center -> 15
    design = box_behnken(_factors(3))
    assert design.n_runs == 15


def test_box_behnken_levels_are_minus1_0_1():
    design = box_behnken(_factors(3))
    coded = design.coded().to_numpy()
    assert set(np.unique(coded)) == {-1.0, 0.0, 1.0}


def test_box_behnken_has_no_corner_points():
    design = box_behnken(_factors(3))
    coded = design.coded().to_numpy()
    # no run sits at an extreme on every factor simultaneously (that's a CCD/factorial corner)
    all_extreme = np.all(np.abs(coded) == 1.0, axis=1)
    assert not all_extreme.any()


def test_box_behnken_center_count():
    design = box_behnken(_factors(3), center=5)
    assert design.n_runs == 12 + 5
    assert design.n_center == 5


def test_box_behnken_requires_three_factors():
    with pytest.raises(ValueError):
        box_behnken(_factors(2))


def test_box_behnken_k5_run_count():
    # k=5: all C(5,2)=10 pairs x 4 edge runs = 40 (canonical BBD), plus 3 center -> 43
    design = box_behnken(_factors(5))
    assert design.n_runs == 40 + 3


def test_box_behnken_rejects_six_or_more_factors():
    # k>=6: all-pairs would give a larger, non-canonical (non-rotatable) design, so reject it
    with pytest.raises(ValueError, match="at most 5 factors"):
        box_behnken(_factors(6))
