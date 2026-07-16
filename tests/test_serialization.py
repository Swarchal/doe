"""JSON round-tripping for FactorSet and Design.

The contract these tests pin down: ``Design.from_dict(d.to_dict())`` reproduces the run
table, factor set, point types, run order, center-point metadata, and -- because coded
values and the model matrix are derived from the factor set -- fitted model behavior.
``to_dict`` output must also survive a real ``json.dumps``/``json.loads`` cycle.
"""

import json

import numpy as np
import pandas as pd
import pytest

from doe.analysis.fit import fit_ols
from doe.analysis.optimize import ResponseGoal
from doe.design import Design
from doe.factors import (
    CategoricalFactor,
    ContinuousFactor,
    FactorSet,
    MixtureFactor,
    factor_from_dict,
)
from doe.generators.factorial import fractional_factorial, full_factorial
from doe.generators.rsm import central_composite
from doe.serialization import ValidationError, json_safe, validate_design_dict

# --------------------------------------------------------------------------- #
# Factors
# --------------------------------------------------------------------------- #


def test_continuous_factor_round_trips():
    f = ContinuousFactor("temp", 20.0, 80.0, units="C")
    restored = ContinuousFactor.from_dict(f.to_dict())
    assert restored == f
    assert f.to_dict()["type"] == "continuous"


def test_categorical_factor_round_trips():
    f = CategoricalFactor("buffer", ("A", "B", "C"), units=None)
    restored = CategoricalFactor.from_dict(f.to_dict())
    assert restored == f
    assert isinstance(restored, CategoricalFactor)
    assert restored.levels == ("A", "B", "C")


def test_factor_dict_is_json_serializable():
    f = ContinuousFactor("p", 0.0, 1.0)
    assert json.loads(json.dumps(f.to_dict())) == f.to_dict()


@pytest.mark.parametrize(
    "factor",
    [
        # bounds/levels derived from an array arrive as numpy scalars; np.integer and
        # np.bool_ subclass nothing json recognises, so to_dict must coerce them
        ContinuousFactor("temp", np.int64(40), np.int64(80), units="C"),
        ContinuousFactor("temp", np.float64(40.0), np.float64(80.0)),
        MixtureFactor("A", np.float64(0.1), np.float64(0.8)),
        CategoricalFactor("buffer", tuple(np.unique(np.array([2, 1])))),
        CategoricalFactor("flag", (np.bool_(True), np.bool_(False))),
    ],
    ids=[
        "continuous-int64",
        "continuous-float64",
        "mixture-float64",
        "levels-int64",
        "levels-bool",
    ],
)
def test_factor_dict_from_numpy_scalars_is_json_serializable(factor):
    payload = factor.to_dict()
    restored = factor_from_dict(json.loads(json.dumps(payload)))
    assert restored == factor


def test_categorical_levels_are_native_python_types():
    f = CategoricalFactor("buffer", tuple(np.unique(np.array([1, 2]))))
    levels = f.to_dict()["levels"]
    assert [type(level) for level in levels] == [int, int]


def test_design_with_numpy_derived_factors_is_json_serializable():
    # the end-to-end path a protocol generator consumes: factors built from array
    # bounds/levels, serialized whole and parsed by a strict JSON reader
    column = np.array([40, 80])
    factors = FactorSet(
        [
            ContinuousFactor("temp", column.min(), column.max()),
            CategoricalFactor("buffer", tuple(np.unique(np.array(["B", "A"])))),
        ]
    )
    runs = pd.DataFrame({"temp": [40, 80], "buffer": ["A", "B"]})
    design = Design(runs, factors, name="numpy-bounds")
    doc = json.loads(json.dumps(design.to_dict()))
    validate_design_dict(doc)
    assert Design.from_dict(doc).runs.equals(design.runs)


def test_factor_set_preserves_order():
    factors = [
        ContinuousFactor("a", 0.0, 1.0),
        CategoricalFactor("cat", ("x", "y")),
        ContinuousFactor("b", -1.0, 2.0, units="m"),
    ]
    fs = FactorSet(factors)
    restored = FactorSet.from_dict(fs.to_dict())
    assert restored.names == fs.names
    assert list(restored) == list(fs)


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #


def _mixed_design() -> Design:
    factors = [
        ContinuousFactor("temp", 0.0, 100.0, units="C"),
        CategoricalFactor("buffer", ("A", "B")),
    ]
    return full_factorial(factors)


def test_design_round_trips_runs_and_factors():
    design = _mixed_design()
    restored = Design.from_dict(design.to_dict())

    assert restored.name == design.name
    assert restored.factors.names == design.factors.names
    pd.testing.assert_frame_equal(restored.runs, design.runs)
    pd.testing.assert_frame_equal(restored.coded(), design.coded())


def test_design_dict_is_json_serializable():
    design = _mixed_design()
    blob = json.dumps(design.to_dict())
    restored = Design.from_dict(json.loads(blob))
    pd.testing.assert_frame_equal(restored.runs, design.runs)


def test_design_round_trips_point_types_and_center_metadata():
    design = central_composite(
        [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)],
        center=4,
    )
    restored = Design.from_dict(json.loads(json.dumps(design.to_dict())))

    assert restored.point_types == design.point_types
    assert restored.n_center == design.n_center
    assert np.array_equal(restored.center_indices, design.center_indices)


def test_design_round_trips_randomized_run_order_and_seed():
    design = central_composite(
        [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)],
        center=3,
    ).randomize(seed=11)

    restored = Design.from_dict(json.loads(json.dumps(design.to_dict())))

    # the std_order column (and its order) survives, and so does the recorded seed
    pd.testing.assert_frame_equal(restored.runs, design.runs)
    assert restored.meta["random_seed"] == 11
    assert restored.meta["randomized"] is True


def test_design_round_trip_preserves_fit_behavior():
    # the real point of keeping natural values + factor set: analysis is unchanged.
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    design = full_factorial(factors, levels=2)
    coded = design.coded().to_numpy()
    y = 10.0 + 3.0 * coded[:, 0] + 2.0 * coded[:, 1] + 1.5 * coded[:, 0] * coded[:, 1]

    restored = Design.from_dict(json.loads(json.dumps(design.to_dict())))
    before = fit_ols(design, y, model="linear")
    after = fit_ols(restored, y, model="linear")

    assert after.term_names == before.term_names
    assert np.allclose(after.coefficients, before.coefficients)


def test_design_round_trips_appended_response_column():
    # responses are stored as extra columns on runs; they must survive serialization too.
    design = _mixed_design()
    with_response = Design(
        design.runs.assign(yield_pct=[10.0, 20.0, 30.0, 40.0]),
        design.factors,
        design.name,
    )
    restored = Design.from_dict(json.loads(json.dumps(with_response.to_dict())))
    pd.testing.assert_frame_equal(restored.runs, with_response.runs)
    assert "yield_pct" in restored.runs.columns


def test_generator_spec_survives_json_and_regenerates_design():
    # the design-spec layer: the document carries the generating *request* (name +
    # parameters) in meta, so the intended experiment -- not just the frozen run
    # table -- can be reconstructed from JSON alone.
    factors = [ContinuousFactor(name, 0.0, 10.0) for name in "abcd"]
    design = fractional_factorial(factors, ["D=ABC"])

    d = json.loads(json.dumps(design.to_dict()))
    spec = d["meta"]["generator"]
    rebuilt = fractional_factorial(
        [factor_from_dict(fd) for fd in d["factors"]], **spec["parameters"]
    )
    pd.testing.assert_frame_equal(rebuilt.runs, design.runs)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _ccd() -> Design:
    return central_composite(
        [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)],
        alpha="rotatable",
        center=3,
    )


def test_validate_accepts_real_designs():
    # both a mixed factorial and a response-surface design should validate cleanly
    validate_design_dict(_mixed_design().to_dict())
    validate_design_dict(_ccd().to_dict())


def test_validate_rejects_unsupported_schema_version():
    d = _mixed_design().to_dict()
    d["schema_version"] = "99.0"
    with pytest.raises(ValidationError, match="schema_version"):
        validate_design_dict(d)


def test_validate_rejects_missing_factor_value():
    d = _mixed_design().to_dict()
    del d["runs"][0]["temp"]
    with pytest.raises(ValidationError, match="temp"):
        validate_design_dict(d)


def test_validate_rejects_invalid_categorical_level():
    d = _mixed_design().to_dict()
    d["runs"][0]["buffer"] = "Z"
    with pytest.raises(ValidationError, match="not a declared level"):
        validate_design_dict(d)


def test_validate_rejects_point_types_length_mismatch():
    d = _ccd().to_dict()
    d["point_types"] = d["point_types"][:-1]
    with pytest.raises(ValidationError, match="point_types"):
        validate_design_dict(d)


def test_validate_rejects_empty_factor_list():
    d = _mixed_design().to_dict()
    d["factors"] = []
    with pytest.raises(ValidationError, match="non-empty list"):
        validate_design_dict(d)


def test_validate_allows_axial_extrapolation_by_default():
    # a rotatable CCD's axial points sit outside [low, high]; default validation accepts them
    d = _ccd().to_dict()
    a_values = [run["a"] for run in d["runs"]]
    assert min(a_values) < 0.0 or max(a_values) > 10.0  # genuinely out of the box
    validate_design_dict(d)  # no raise


def test_validate_check_ranges_flags_extrapolation():
    d = _ccd().to_dict()
    with pytest.raises(ValidationError, match="outside"):
        validate_design_dict(d, check_ranges=True)


def test_validate_collects_multiple_errors():
    d = _mixed_design().to_dict()
    del d["runs"][0]["temp"]
    d["runs"][1]["buffer"] = "Z"
    with pytest.raises(ValidationError) as exc:
        validate_design_dict(d)
    assert len(exc.value.errors) >= 2


# --- FactorSet cross-factor invariants (mirror FactorSet, so Design.from_dict never
#     raises a bare ValueError on a document validate_design_dict accepted) --------------


def _mixture_document(*components, runs=None):
    factors = [
        {"type": "mixture", "name": name, "low": low, "high": high}
        for name, low, high in components
    ]
    return {
        "schema_version": "1.0",
        "factors": factors,
        "runs": runs if runs is not None else [{name: 0.0 for name, *_ in components}],
    }


def test_validate_rejects_mixture_mixed_with_other_factor_types():
    document = {
        "schema_version": "1.0",
        "factors": [
            {"type": "mixture", "name": "A", "low": 0.0, "high": 1.0},
            {"type": "continuous", "name": "T", "low": 20.0, "high": 80.0},
        ],
        "runs": [{"A": 0.5, "T": 50.0}],
    }
    with pytest.raises(ValidationError, match="combined with other factor types"):
        validate_design_dict(document)


def test_validate_rejects_a_lone_mixture_component():
    with pytest.raises(ValidationError, match="at least 2 components"):
        validate_design_dict(_mixture_document(("A", 0.0, 1.0), runs=[{"A": 1.0}]))


def test_validate_rejects_an_infeasible_mixture_blend():
    # sum(low) = 1.2 > 1, so no blend of A + B can sum to 1
    document = _mixture_document(("A", 0.6, 0.9), ("B", 0.6, 0.9), runs=[{"A": 0.5, "B": 0.5}])
    with pytest.raises(ValidationError, match="feasible blend"):
        validate_design_dict(document)


def test_validate_accepts_a_feasible_all_mixture_design():
    # regression: a well-formed mixture design must still validate cleanly
    document = _mixture_document(
        ("A", 0.0, 1.0), ("B", 0.0, 1.0), ("C", 0.0, 1.0),
        runs=[{"A": 1.0, "B": 0.0, "C": 0.0}],
    )
    validate_design_dict(document)  # no raise


def test_validate_guards_documents_that_Design_from_dict_would_reject():
    # the whole point: anything validate accepts, Design.from_dict builds; anything the
    # FactorSet would reject, validate now catches first (no bare ValueError escapes).
    bad = _mixture_document(("A", 0.6, 0.9), ("B", 0.6, 0.9), runs=[{"A": 0.5, "B": 0.5}])
    with pytest.raises(ValidationError):
        validate_design_dict(bad)
    with pytest.raises(ValueError):  # Design.from_dict -> FactorSet still raises, as before
        Design.from_dict(bad)


# --------------------------------------------------------------------------- #
# ResponseGoal as data
# --------------------------------------------------------------------------- #


def _quadratic_fit():
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    design = central_composite(factors, center=3)
    coded = design.coded().to_numpy()
    y = 50.0 - (coded[:, 0] ** 2 + coded[:, 1] ** 2)
    return fit_ols(design, y, model="quadratic")


def test_response_goal_round_trips_definition():
    result = _quadratic_fit()
    goal = ResponseGoal(result, goal="target", low=40.0, high=50.0, target=48.0, weight=2.0)

    data = json.loads(json.dumps(goal.to_dict()))
    restored = ResponseGoal.from_dict(data, result)

    assert restored.goal == "target"
    assert (restored.low, restored.high, restored.target, restored.weight) == (
        40.0,
        50.0,
        48.0,
        2.0,
    )
    # the reconstructed goal scores responses identically
    assert restored.desirability(45.0) == goal.desirability(45.0)


def test_response_goal_to_dict_omits_derived_fit_result():
    result = _quadratic_fit()
    goal = ResponseGoal(result, goal="max", low=0.0, high=1.0)
    d = goal.to_dict()
    assert "result" not in d
    assert set(d) == {"goal", "low", "high", "target", "weight"}


def test_response_goal_from_dict_rejects_unknown_goal():
    result = _quadratic_fit()
    with pytest.raises(ValueError, match="unknown goal"):
        ResponseGoal.from_dict({"goal": "bogus", "low": 0.0, "high": 1.0}, result)


# --------------------------------------------------------------------------- #
# json_safe -- the shared "JSON has no NaN" coercion every to_dict routes through
# --------------------------------------------------------------------------- #


def test_json_safe_coerces_numpy_scalars():
    assert json_safe(np.float64(1.5)) == 1.5
    assert isinstance(json_safe(np.float64(1.5)), float)
    assert json_safe(np.int64(3)) == 3
    assert isinstance(json_safe(np.int64(3)), int)
    assert json_safe(np.bool_(True)) is True


def test_json_safe_coerces_numpy_arrays_to_nested_lists():
    assert json_safe(np.array([1, 2, 3])) == [1, 2, 3]
    assert json_safe(np.array([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]


def test_json_safe_maps_non_finite_floats_to_none():
    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe(float("-inf")) is None
    assert json_safe(np.nan) is None
    assert json_safe(np.float64("inf")) is None
    assert json_safe(1.0) == 1.0


def test_json_safe_walks_nested_mappings_and_sequences():
    payload = {
        "a": np.nan,
        "b": [np.int64(2), float("inf"), {"c": np.array([1.0, np.nan])}],
        "d": (np.bool_(False), "text"),
    }
    assert json_safe(payload) == {
        "a": None,
        "b": [2, None, {"c": [1.0, None]}],
        "d": [False, "text"],
    }


def test_json_safe_passes_through_ordinary_values():
    assert json_safe("text") == "text"
    assert json_safe(True) is True
    assert json_safe(None) is None
    assert json_safe(42) == 42


def test_json_safe_output_survives_json_dumps():
    payload = {"a": np.float64(1.5), "b": [np.nan, np.int64(2)]}
    dumped = json.loads(json.dumps(json_safe(payload)))
    assert dumped == {"a": 1.5, "b": [None, 2]}


# --- whole_plots + hard_to_change serialization (Phase 5b) ----------------------------------


def test_whole_plots_round_trip_and_omitted_when_none():
    factors = FactorSet(
        [ContinuousFactor("oven", 200, 400, hard_to_change=True), ContinuousFactor("time", 5, 15)]
    )
    runs = pd.DataFrame({"oven": [200, 200, 300, 300], "time": [5, 15, 5, 15]})
    d = Design(runs, factors, whole_plots=(0, 0, 1, 1))
    payload = d.to_dict()
    assert payload["whole_plots"] == [0, 0, 1, 1]
    assert payload["factors"][0]["hard_to_change"] is True
    validate_design_dict(payload)
    restored = Design.from_dict(json.loads(json.dumps(payload)))
    assert restored.whole_plots == (0, 0, 1, 1)
    assert restored.factors["oven"].hard_to_change is True
    # a plain design emits no whole_plots key
    assert "whole_plots" not in Design(runs, factors).to_dict()


def test_validate_rejects_misaligned_whole_plots():
    factors = FactorSet([ContinuousFactor("x", 0, 1)])
    payload = Design(pd.DataFrame({"x": [0, 1]}), factors).to_dict()
    payload["whole_plots"] = [0, 0, 0]
    with pytest.raises(ValidationError, match="whole_plots"):
        validate_design_dict(payload)


def test_booleans_are_not_accepted_as_continuous_bounds():
    # bool subclasses int, so a naive isinstance(v, int | float) would validate {"low": false,
    # "high": true} as a well-formed continuous factor spanning [0, 1]. JSON booleans are not
    # numbers.
    document = {
        "schema_version": "1.0",
        "name": "",
        "factors": [{"type": "continuous", "name": "a", "low": False, "high": True}],
        "runs": [{"a": 0.5}],
        "point_types": None,
        "meta": {},
    }
    with pytest.raises(ValidationError, match="must be numbers"):
        validate_design_dict(document)


def test_boolean_run_value_is_not_accepted_as_numeric():
    document = {
        "schema_version": "1.0",
        "name": "",
        "factors": [{"type": "continuous", "name": "a", "low": 0.0, "high": 1.0}],
        "runs": [{"a": True}],
        "point_types": None,
        "meta": {},
    }
    with pytest.raises(ValidationError, match="not numeric"):
        validate_design_dict(document)
