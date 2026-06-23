"""The :class:`Design` container -- replication and run bookkeeping.

Written test-first. ``replicate`` is the first-class way to repeat a design's runs
(replacing ad-hoc DataFrame stacking), keeping the run<->condition mapping and the
``point_types`` labels that lack-of-fit depends on in lock-step.
"""

import numpy as np
import pytest

from doe import fit_ols, full_factorial
from doe.factors import ContinuousFactor
from doe.generators.rsm import central_composite


def _factorial_2x2():
    a = ContinuousFactor("a", 0.0, 10.0)
    b = ContinuousFactor("b", 0.0, 10.0)
    return full_factorial([a, b], levels=2)


# --------------------------------------------------------------------------- #
# replicate -- run counts and ordering
# --------------------------------------------------------------------------- #


def test_replicate_scales_run_count():
    design = _factorial_2x2()
    assert design.replicate(3).n_runs == 4 * 3


def test_replicate_default_tiles_whole_design():
    design = _factorial_2x2()
    rep = design.replicate(3)  # each=False (default): [design, design, design]
    base = design.runs.to_numpy()
    out = rep.runs.to_numpy()
    # three consecutive whole-design passes
    assert np.allclose(out[0:4], base)
    assert np.allclose(out[4:8], base)
    assert np.allclose(out[8:12], base)


def test_replicate_each_groups_replicates_by_condition():
    design = _factorial_2x2()
    rep = design.replicate(3, each=True)  # row0,row0,row0, row1,row1,row1, ...
    base = design.runs.to_numpy()
    out = rep.runs.to_numpy()
    for i in range(4):
        assert np.allclose(out[3 * i : 3 * i + 3], base[i])


def test_replicate_n_one_is_unchanged():
    design = _factorial_2x2()
    rep = design.replicate(1)
    assert rep.n_runs == design.n_runs
    assert np.allclose(rep.runs.to_numpy(), design.runs.to_numpy())


def test_replicate_rejects_non_positive():
    design = _factorial_2x2()
    with pytest.raises(ValueError):
        design.replicate(0)
    with pytest.raises(ValueError):
        design.replicate(-2)


# --------------------------------------------------------------------------- #
# replicate -- point_types / center bookkeeping carried along
# --------------------------------------------------------------------------- #


def test_replicate_carries_point_types_and_scales_centers():
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = central_composite(factors, center=4)
    rep = design.replicate(2)
    assert rep.point_types is not None
    assert len(rep.point_types) == rep.n_runs
    # center replicates double, and remain correctly indexed
    assert rep.n_center == design.n_center * 2
    centers = rep.runs.iloc[rep.center_indices][rep.factors.names].to_numpy()
    assert np.allclose(centers, 5.0)


def test_replicate_without_point_types_stays_none():
    design = _factorial_2x2()  # generators that don't track point types
    assert design.point_types is None
    assert design.replicate(2).point_types is None


def test_replicate_records_count_and_preserves_factors():
    design = _factorial_2x2()
    rep = design.replicate(3)
    assert rep.meta["replicates"] == 3
    assert rep.factors is design.factors


# --------------------------------------------------------------------------- #
# replicate -- the analysis path it exists to support
# --------------------------------------------------------------------------- #


def test_replicate_each_aligns_with_condition_grouped_response():
    # responses measured three-per-condition (triplicates consecutive) line up with each=True
    design = _factorial_2x2()
    rep = design.replicate(3, each=True)
    # y = 50 + 10*x1 (coded) + noise-free; main effect of 'a' should come back as 20 (2*coef)
    coded = rep.coded().to_numpy()
    y = 50.0 + 10.0 * coded[:, 0]
    result = fit_ols(rep, y, model="linear")
    assert np.isclose(dict(result.summary())["a"][1], 20.0)
