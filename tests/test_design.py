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
    assert np.isclose(result.summary().loc["a", "effect"], 20.0)


def test_randomize_can_be_repeated_without_duplicating_std_order():
    design = central_composite(
        [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)],
        center=3,
    )

    randomized = design.randomize(seed=1).randomize(seed=2)

    assert list(randomized.runs.columns).count("std_order") == 1
    assert list(randomized.runs.columns)[0] == "std_order"
    assert sorted(randomized.runs["std_order"].tolist()) == list(range(design.n_runs))


def test_randomize_records_explicit_seed():
    design = _factorial_2x2()
    randomized = design.randomize(seed=42)
    assert randomized.meta["random_seed"] == 42
    assert randomized.meta["randomized"] is True


def test_randomize_records_a_concrete_seed_when_none_given():
    # passing no seed must still record a concrete, reproducible seed (the serialization
    # gap this fixes: meta previously said "randomized" but not *how*).
    design = _factorial_2x2()
    randomized = design.randomize()
    seed = randomized.meta["random_seed"]
    assert isinstance(seed, int)
    # re-running with the recorded seed reproduces the exact shuffle
    replayed = design.randomize(seed=seed)
    assert randomized.runs["std_order"].tolist() == replayed.runs["std_order"].tolist()


def test_randomize_seed_reproduces_run_order():
    design = central_composite(
        [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)],
        center=3,
    )
    a = design.randomize(seed=7)
    b = design.randomize(seed=7)
    assert a.runs["std_order"].tolist() == b.runs["std_order"].tolist()
    assert a.runs[["a", "b"]].equals(b.runs[["a", "b"]])


# --------------------------------------------------------------------------- #
# with_response -- attaching a measured response as a column
# --------------------------------------------------------------------------- #


def test_with_response_appends_column_without_mutating_original():
    design = _factorial_2x2()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    withy = design.with_response("yield", y)
    assert "yield" not in design.runs.columns  # original untouched
    assert withy.runs["yield"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert withy.factors is design.factors


def test_with_response_rejects_length_mismatch():
    design = _factorial_2x2()
    with pytest.raises(ValueError, match="3 values but there are 4 runs"):
        design.with_response("y", np.array([1.0, 2.0, 3.0]))


def test_with_response_rejects_factor_name_collision():
    design = _factorial_2x2()
    with pytest.raises(ValueError, match="collides with a factor column"):
        design.with_response("a", np.zeros(4))


def test_with_response_rejects_non_1d():
    design = _factorial_2x2()
    with pytest.raises(ValueError, match="one-dimensional"):
        design.with_response("y", np.zeros((4, 2)))


def test_fit_ols_accepts_response_column_name():
    design = _factorial_2x2().replicate(2)  # residual dof > 0, no saturation warning
    coded = design.coded().to_numpy()
    y = 50.0 + 10.0 * coded[:, 0]
    by_name = fit_ols(design.with_response("y", y), "y", model="linear")
    by_array = fit_ols(design, y, model="linear")
    assert np.allclose(by_name.coefficients, by_array.coefficients)


def test_fit_ols_unknown_response_column_raises():
    design = _factorial_2x2()
    with pytest.raises(ValueError, match="no response column 'missing'"):
        fit_ols(design, "missing")


def test_response_survives_replicate_and_randomize():
    design = _factorial_2x2()
    y = np.array([10.0, 20.0, 30.0, 40.0])
    withy = design.with_response("y", y)

    # each=True groups a condition's replicates, so the response repeats in place
    rep = withy.replicate(2, each=True)
    assert rep.runs["y"].tolist() == [10, 10, 20, 20, 30, 30, 40, 40]

    # randomize reindexes every column together, so y stays glued to its run.
    # a==0 & b==0 is the run whose response is 10.0; find it wherever it shuffled to.
    rand = withy.randomize(seed=3)
    match = rand.runs[(rand.runs["a"] == 0.0) & (rand.runs["b"] == 0.0)]
    assert match["y"].tolist() == [10.0]


# --------------------------------------------------------------------------- #
# with_responses -- attaching several measured responses in one call
# --------------------------------------------------------------------------- #


def test_with_responses_attaches_all_columns_aligned():
    design = _factorial_2x2()
    yld = np.array([1.0, 2.0, 3.0, 4.0])
    impurity = np.array([0.1, 0.2, 0.3, 0.4])
    both = design.with_responses(yield_=yld, impurity=impurity)
    assert both.runs["yield_"].tolist() == yld.tolist()
    assert both.runs["impurity"].tolist() == impurity.tolist()


def test_with_responses_does_not_mutate_original():
    design = _factorial_2x2()
    design.with_responses(yield_=np.zeros(4), impurity=np.ones(4))
    assert "yield_" not in design.runs.columns
    assert "impurity" not in design.runs.columns


def test_with_responses_rejects_length_mismatch():
    design = _factorial_2x2()
    with pytest.raises(ValueError, match="3 values but there are 4 runs"):
        design.with_responses(yield_=np.zeros(4), impurity=np.zeros(3))


def test_with_responses_rejects_factor_name_collision():
    design = _factorial_2x2()
    with pytest.raises(ValueError, match="collides with a factor column"):
        design.with_responses(a=np.zeros(4))


def test_with_responses_rejects_empty_call():
    design = _factorial_2x2()
    with pytest.raises(ValueError, match="no responses given"):
        design.with_responses()


def test_with_responses_accepts_non_identifier_name_via_unpacking():
    design = _factorial_2x2()
    withy = design.with_responses(**{"yield %": np.array([1.0, 2.0, 3.0, 4.0])})
    assert withy.runs["yield %"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_with_responses_preserves_point_types_and_meta():
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = central_composite(factors, center=4)
    withy = design.with_responses(yield_=np.zeros(design.n_runs), impurity=np.ones(design.n_runs))
    assert withy.point_types == design.point_types
    assert withy.meta == design.meta
    assert withy.factors is design.factors
