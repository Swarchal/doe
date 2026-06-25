"""Validation for serialized design documents.

Structural checks that catch a malformed or internally inconsistent ``Design.to_dict()``
document *before* it is executed on a robot or fed to analysis -- the kind of corruption
hand edits, other tools, or an older schema version can introduce. Kept dependency-free
(no external JSON Schema library) so the core library stays on the scipy stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from .design import SCHEMA_VERSION


class ValidationError(ValueError):
    """Raised when a serialized design document is structurally invalid.

    Carries the full list of problems found (``.errors``), not just the first, so a
    malformed document can be fixed in a single pass.
    """

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        joined = "\n  - ".join(self.errors)
        super().__init__(f"invalid design document:\n  - {joined}")


def _is_list(value: object) -> TypeGuard[Sequence[Any]]:
    """A JSON array: a sequence that is not a string/bytes scalar."""
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _supported_major(version: object) -> bool:
    """Accept any document whose major schema version matches this build's."""
    if not isinstance(version, str):
        return False
    return version.split(".")[0] == SCHEMA_VERSION.split(".")[0]


def validate_design_dict(data: Mapping[str, Any], *, check_ranges: bool = False) -> None:
    """Validate a :meth:`doe.Design.to_dict` document, raising on any problem.

    Checks the document is structurally sound and internally consistent:

    * a supported ``schema_version`` (same major version as this build);
    * a non-empty, well-formed, uniquely-named ``factors`` list (continuous factors need
      ``high > low``; categorical factors need >= 2 unique ``levels``);
    * every run carrying a value for every declared factor;
    * categorical run values drawn from the declared ``levels``;
    * continuous run values numeric;
    * ``point_types`` (when present) aligned one-to-one with the runs;
    * ``meta`` (when present) a mapping.

    Continuous values are **not** range-checked by default: response-surface designs
    (central composite) deliberately place axial points outside the ``[low, high]`` factor
    box, so an out-of-range value is normal there, not an error. Pass ``check_ranges=True``
    to additionally require every continuous value to lie within its factor bounds -- useful
    for hand-authored factorial plans where no extrapolation is expected.

    Raises :class:`ValidationError` whose ``.errors`` lists every problem found.
    """
    if not isinstance(data, Mapping):
        raise ValidationError([f"document must be a mapping, got {type(data).__name__}"])

    errors: list[str] = []

    version = data.get("schema_version")
    if version is None:
        errors.append("missing 'schema_version'")
    elif not _supported_major(version):
        errors.append(
            f"unsupported schema_version {version!r}; this build understands {SCHEMA_VERSION!r}"
        )

    factor_names: list[str] = []
    categorical_levels: dict[str, set[object]] = {}
    continuous_bounds: dict[str, tuple[float, float]] = {}

    raw_factors = data.get("factors")
    if _is_list(raw_factors) and raw_factors:
        factor_records: list[Any] = list(raw_factors)
    else:
        errors.append("'factors' must be a non-empty list")
        factor_records = []
    for i, fd in enumerate(factor_records):
        if not isinstance(fd, Mapping):
            errors.append(f"factor[{i}] must be a mapping")
            continue
        raw_name = fd.get("name")
        name = raw_name if isinstance(raw_name, str) and raw_name else None
        if name is None:
            errors.append(f"factor[{i}] missing a non-empty string 'name'")
        kind = fd.get("type")
        if kind == "continuous":
            low, high = fd.get("low"), fd.get("high")
            if not isinstance(low, int | float) or not isinstance(high, int | float):
                errors.append(f"factor {name!r}: continuous 'low'/'high' must be numbers")
            elif high <= low:
                errors.append(f"factor {name!r}: high ({high}) must exceed low ({low})")
            elif name is not None:
                continuous_bounds[name] = (float(low), float(high))
        elif kind == "categorical":
            levels = fd.get("levels")
            if not _is_list(levels) or len(levels) < 2:
                errors.append(
                    f"factor {name!r}: categorical 'levels' must list at least 2 values"
                )
            else:
                if len(set(levels)) != len(levels):
                    errors.append(f"factor {name!r}: categorical levels must be unique")
                if name is not None:
                    categorical_levels[name] = set(levels)
        else:
            errors.append(f"factor {name!r}: unknown type {kind!r}")
        if name is not None:
            factor_names.append(name)

    dupes = sorted({n for n in factor_names if factor_names.count(n) > 1})
    if dupes:
        errors.append(f"duplicate factor names: {dupes}")

    raw_runs = data.get("runs")
    if _is_list(raw_runs):
        run_records: list[Any] = list(raw_runs)
    else:
        errors.append("'runs' must be a list")
        run_records = []
    for i, record in enumerate(run_records):
        if not isinstance(record, Mapping):
            errors.append(f"run[{i}] must be a mapping")
            continue
        for fname in factor_names:
            if fname not in record:
                errors.append(f"run[{i}] missing value for factor {fname!r}")
                continue
            value = record[fname]
            if fname in categorical_levels and value not in categorical_levels[fname]:
                errors.append(f"run[{i}] factor {fname!r}: {value!r} is not a declared level")
            elif fname in continuous_bounds:
                if not isinstance(value, int | float):
                    errors.append(f"run[{i}] factor {fname!r}: {value!r} is not numeric")
                elif check_ranges:
                    lo, hi = continuous_bounds[fname]
                    if not lo <= value <= hi:
                        errors.append(
                            f"run[{i}] factor {fname!r}: {value} outside [{lo}, {hi}]"
                        )

    point_types = data.get("point_types")
    if point_types is not None:
        if not _is_list(point_types):
            errors.append("'point_types' must be a list or null")
        elif len(point_types) != len(run_records):
            errors.append(
                f"'point_types' has {len(point_types)} entries but there are "
                f"{len(run_records)} runs"
            )

    meta = data.get("meta")
    if meta is not None and not isinstance(meta, Mapping):
        errors.append("'meta' must be a mapping or null")

    if errors:
        raise ValidationError(errors)
