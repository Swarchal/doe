"""The JSON coercion rule, in a leaf module.

Lives below :mod:`doe.design` and :mod:`doe.serialization` so both can import it without
a cycle (``serialization`` already imports ``design`` for ``SCHEMA_VERSION``). The public
home of :func:`json_safe` is :mod:`doe.serialization`, which re-exports it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively coerce ``value`` into something :func:`json.dumps` can serialize.

    The single place the "JSON has no NaN" rule lives: every ``to_dict`` in the library
    routes its output through this before returning, so no caller has to sanitize a
    result by hand. Three coercions, applied recursively:

    * numpy scalars (``np.integer``/``np.floating``/``np.bool_``/...) and ``np.ndarray``
      become native Python types (``ndarray`` -> nested ``list``);
    * non-finite floats (``NaN``/``+-Infinity``, numpy or native) become ``None`` -- JSON
      has no literal for them, and this is the encoding the web service's response
      contract (``docs/WEBSERVICE_API.md``) standardises on for undefined statistics
      (saturated-model standard errors, Scheffé effects, undefined F/p);
    * mappings and sequences (other than ``str``/``bytes``) are walked so a nested
      structure -- a dict of arrays, a list of term records -- comes out fully
      JSON-safe in one pass.

    Anything else (``str``, ``bool``, ``int``, ``None``, an already-finite ``float``)
    passes through unchanged.

    Examples:
        >>> import numpy as np
        >>> json_safe(np.float64(1.5))
        1.5
        >>> json_safe(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> json_safe(float("nan")) is None
        True
        >>> json_safe({"a": np.nan, "b": [np.int64(2), float("inf")]})
        {'a': None, 'b': [2, None]}
    """
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [json_safe(item) for item in value]
    return value
