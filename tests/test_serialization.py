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
from doe.factors import CategoricalFactor, ContinuousFactor, FactorSet, factor_from_dict
from doe.generators.factorial import fractional_factorial, full_factorial
from doe.generators.rsm import central_composite
from doe.serialization import ValidationError, validate_design_dict

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
