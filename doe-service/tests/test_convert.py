"""Unit tests for the Milestone 1 request/library plumbing (``doe_service.convert``)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest

from doe import ContinuousFactor, FactorSet, ValidationError, full_factorial
from doe.analysis.fit import SaturatedFitWarning
from doe_service.convert import (
    call_library,
    captured_warnings,
    check_factor_count,
    check_run_count,
    check_search_budget,
    design_from_document,
    jsonable,
    region_array,
    resolve_model,
)
from doe_service.errors import Infeasible
from doe_service.limits import LimitExceeded, Limits
from doe_service.schemas.common import ModelSpecObject


def _factors() -> FactorSet:
    return FactorSet(
        [
            ContinuousFactor("temp", low=20.0, high=80.0, units="C"),
            ContinuousFactor("time", low=0.0, high=10.0),
        ]
    )


# --------------------------------------------------------------------------- #
# design_from_document
# --------------------------------------------------------------------------- #


def test_design_from_document_round_trips_a_valid_document() -> None:
    design = full_factorial(_factors())
    document = design.to_dict()

    restored = design_from_document(document)

    assert restored.factors.names == design.factors.names
    assert restored.runs.to_dict(orient="records") == design.runs.to_dict(orient="records")


def test_design_from_document_raises_validation_error_with_all_problems() -> None:
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "name": "bad",
        "factors": [{"type": "continuous", "name": "temp", "low": 20.0, "high": 80.0}],
        "runs": [{}, {"temp": 50.0}],  # run 0 missing 'temp'
        "point_types": ["factorial"],  # 1 entry for 2 runs
        "meta": {},
    }

    with pytest.raises(ValidationError) as excinfo:
        design_from_document(document)

    errors = excinfo.value.errors
    assert len(errors) >= 2
    assert any("temp" in e and "missing" in e for e in errors)
    assert any("point_types" in e for e in errors)


# --------------------------------------------------------------------------- #
# resolve_model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,expected",
    [
        ("linear", (1, True)),
        ("quadratic", (2, True)),
        ("scheffe-linear", (1, False)),
        ("scheffe-quadratic", (2, False)),
    ],
)
def test_resolve_model_convenience_names(name: str, expected: tuple[int, bool]) -> None:
    assert resolve_model(name, default="linear") == expected


def test_resolve_model_object_form() -> None:
    assert resolve_model({"order": 2, "interactions": False}, default="linear") == (2, False)


def test_resolve_model_object_form_defaults_interactions_true() -> None:
    assert resolve_model({"order": 2}, default="linear") == (2, True)


def test_resolve_model_accepts_model_spec_object_instance() -> None:
    spec = ModelSpecObject(order=2, interactions=False)
    assert resolve_model(spec, default="linear") == (2, False)


def test_resolve_model_falls_back_to_default_when_none() -> None:
    assert resolve_model(None, default="quadratic") == (2, True)


def test_resolve_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        resolve_model("cubic", default="linear")


def test_resolve_model_rejects_object_missing_order() -> None:
    with pytest.raises(ValueError, match="order"):
        resolve_model({"interactions": True}, default="linear")


def test_resolve_model_rejects_bad_order() -> None:
    with pytest.raises(ValueError, match="order"):
        resolve_model({"order": 3}, default="linear")


# --------------------------------------------------------------------------- #
# captured_warnings
# --------------------------------------------------------------------------- #


def test_captured_warnings_maps_saturated_fit_warning_by_category() -> None:
    with captured_warnings() as collected:
        warnings.warn("model is saturated", SaturatedFitWarning, stacklevel=2)

    assert collected == ["saturated_model"]


def test_captured_warnings_falls_back_to_message_for_unknown_categories() -> None:
    with captured_warnings() as collected:
        warnings.warn("something else happened", UserWarning, stacklevel=2)

    assert collected == ["something else happened"]


def test_captured_warnings_records_multiple_warnings_in_order() -> None:
    with captured_warnings() as collected:
        warnings.warn("first", UserWarning, stacklevel=2)
        warnings.warn("second fit was saturated", SaturatedFitWarning, stacklevel=2)

    assert collected == ["first", "saturated_model"]


def test_captured_warnings_empty_when_nothing_warned() -> None:
    with captured_warnings() as collected:
        pass

    assert collected == []


# --------------------------------------------------------------------------- #
# jsonable
# --------------------------------------------------------------------------- #


def test_jsonable_converts_nan_to_none() -> None:
    assert jsonable(float("nan")) is None


def test_jsonable_converts_numpy_scalars() -> None:
    result = jsonable(np.float64(1.5))
    assert result == 1.5
    assert isinstance(result, float)


def test_jsonable_walks_nested_structures() -> None:
    result = jsonable({"a": np.array([1.0, math.inf]), "b": [np.int64(2)]})
    assert result == {"a": [1.0, None], "b": [2]}


# --------------------------------------------------------------------------- #
# call_library
# --------------------------------------------------------------------------- #


def test_call_library_wraps_bare_value_error_as_infeasible() -> None:
    def _raises() -> None:
        raise ValueError("bad sobol size")

    with pytest.raises(Infeasible) as excinfo:
        call_library(_raises)

    assert excinfo.value.message == "bad sobol size"


def test_call_library_lets_doe_validation_error_pass_through() -> None:
    def _raises() -> None:
        raise ValidationError(["problem one", "problem two"])

    with pytest.raises(ValidationError) as excinfo:
        call_library(_raises)

    assert excinfo.value.errors == ["problem one", "problem two"]


def test_call_library_does_not_catch_unrelated_exceptions() -> None:
    def _raises() -> None:
        raise TypeError("wrong argument")

    with pytest.raises(TypeError):
        call_library(_raises)


def test_call_library_returns_the_wrapped_call_result() -> None:
    assert call_library(lambda x, y: x + y, 2, y=3) == 5


# --------------------------------------------------------------------------- #
# limits (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6)
# --------------------------------------------------------------------------- #


def test_check_factor_count_within_cap_is_a_no_op() -> None:
    check_factor_count(list(_factors()), limits=Limits(max_factors=2))


def test_check_factor_count_over_cap_names_the_cap_and_ceiling() -> None:
    with pytest.raises(LimitExceeded, match=r"too many factors: 2 exceeds the cap of 1"):
        check_factor_count(list(_factors()), limits=Limits(max_factors=1))


def test_check_run_count_within_cap_is_a_no_op() -> None:
    check_run_count(5, limits=Limits(max_runs=5))


def test_check_run_count_over_cap_names_the_cap_and_ceiling() -> None:
    with pytest.raises(LimitExceeded, match=r"too many runs: 6 exceeds the cap of 5"):
        check_run_count(6, limits=Limits(max_runs=5))


def test_check_search_budget_within_caps_is_a_no_op() -> None:
    check_search_budget(3, 20, limits=Limits(max_restarts=3, max_iter=20))


def test_check_search_budget_over_restarts_cap_names_the_cap_and_ceiling() -> None:
    with pytest.raises(LimitExceeded, match=r"n_restarts 4 exceeds the cap of 3"):
        check_search_budget(4, 20, limits=Limits(max_restarts=3, max_iter=20))


def test_check_search_budget_over_iter_cap_names_the_cap_and_ceiling() -> None:
    with pytest.raises(LimitExceeded, match=r"max_iter 21 exceeds the cap of 20"):
        check_search_budget(3, 21, limits=Limits(max_restarts=3, max_iter=20))


def test_region_array_none_passes_through() -> None:
    assert region_array(None, n_factors=2) is None


def test_region_array_builds_ndarray_for_a_well_shaped_region() -> None:
    result = region_array([[0.0, 1.0], [1.0, -1.0]], n_factors=2)
    assert result is not None
    assert result.shape == (2, 2)


def test_region_array_shape_mismatch_is_infeasible() -> None:
    with pytest.raises(Infeasible, match="2 coordinate"):
        region_array([[0.0, 1.0], [1.0]], n_factors=2)


def test_region_array_row_count_over_cap_is_limit_exceeded() -> None:
    with pytest.raises(LimitExceeded, match=r"region has 3 rows, exceeding the cap of 2"):
        region_array([[0.0], [0.5], [1.0]], n_factors=1, limits=Limits(max_region_rows=2))


def test_design_from_document_enforces_run_and_factor_count_caps_via_the_defaults() -> None:
    # design_from_document always checks against the real DEFAULT_LIMITS (it has no
    # ``limits`` parameter -- every caller shares one deployment-wide cap); the HTTP-level
    # tests in ``test_limits.py`` exercise this against the default 32-factor/10_000-run
    # ceilings directly. Here we confirm a document comfortably inside both is unaffected.
    design = full_factorial(_factors())
    document = design.to_dict()
    restored = design_from_document(document)
    assert restored.n_runs == design.n_runs
