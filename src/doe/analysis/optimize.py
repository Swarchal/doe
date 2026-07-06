"""Phase 2b: response-surface optimization.

A fitted second-order model is written in coded units as a quadratic form

    y-hat = b0 + x^T b + x^T B x

where ``b`` is the vector of main-effect coefficients and ``B`` is the symmetric matrix
of second-order coefficients (``B[i, i] = coef(x_i^2)``, ``B[i, j] = 1/2 coef(x_i:x_j)``).
From that we derive:

* :func:`stationary_point` -- the unconstrained stationary point ``x_s = -1/2 B^-1 b`` plus
  a canonical (eigenvalue) analysis classifying it as a maximum / minimum / saddle,
* :func:`optimum` -- the constrained optimum over the coded design box (``scipy.optimize``),
  used when the stationary point is infeasible,
* :func:`desirability` -- multi-response Derringer-Suich desirability optimization.

Everything is computed in coded units and decoded to natural units for reporting, matching
the rest of the library (designs are fitted coded, reported natural).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import optimize as sciopt

from ..factors import ContinuousFactor, FactorSet
from .fit import FitResult

Goal = Literal["max", "min", "target"]
Bounds = tuple[float, float] | Sequence[tuple[float, float]]


# --------------------------------------------------------------------------- #
# Quadratic-form extraction and prediction (coded units)
# --------------------------------------------------------------------------- #


def _quadratic_form(result: FitResult) -> tuple[float, np.ndarray, np.ndarray]:
    """Extract ``(b0, b, B)`` so that ``y-hat = b0 + x^T b + x^T B x`` in coded units."""
    # The quadratic form is over the continuous factors' coded coordinates. A categorical
    # factor is expanded into ``factor[level]`` contrast columns whose term names are not
    # factor names, and optimizing a smooth surface over a discrete dimension is not defined,
    # so reject such fits with a clear error rather than a KeyError deep in the parse below.
    non_continuous = [
        name for name in result.factors.names
        if not isinstance(result.factors[name], ContinuousFactor)
    ]
    if non_continuous:
        raise TypeError(
            "surface optimization requires all-continuous factors; "
            f"got categorical factor(s) {non_continuous}"
        )
    names = result.factors.names
    index = {name: i for i, name in enumerate(names)}
    k = len(names)
    b0 = 0.0
    b = np.zeros(k)
    big_b = np.zeros((k, k))
    for term, raw in zip(result.term_names, result.coefficients, strict=True):
        coef = float(raw)
        if term == "Intercept":
            b0 = coef
        elif ":" in term:  # two-factor interaction "a:b"
            a, c = term.split(":")
            i, j = index[a], index[c]
            big_b[i, j] += 0.5 * coef
            big_b[j, i] += 0.5 * coef
        elif "^" in term:  # quadratic term "a^2"
            base, power = term.split("^")
            if int(power) != 2:
                raise ValueError(f"only quadratic (^2) terms are supported, got {term!r}")
            i = index[base]
            big_b[i, i] += coef
        else:  # main effect (a bare factor name)
            b[index[term]] += coef
    return b0, b, big_b


def _predict(b0: float, b: np.ndarray, big_b: np.ndarray, x: np.ndarray) -> float:
    """Evaluate the quadratic form at a single coded point ``x``."""
    return float(b0 + x @ b + x @ big_b @ x)


def _decode(factors: FactorSet, coded: np.ndarray) -> dict[str, float]:
    """Decode a coded point to a ``{factor_name: natural_value}`` mapping."""
    out: dict[str, float] = {}
    for i, name in enumerate(factors.names):
        factor = factors[name]
        if not isinstance(factor, ContinuousFactor):
            raise TypeError(f"factor {name!r} is not continuous; cannot decode")
        out[name] = float(factor.decode(np.asarray(coded[i])))
    return out


def _format_point(natural: dict[str, float]) -> str:
    """Render a ``{factor: natural_value}`` mapping as ``a=1.23, b=4.56`` for reprs."""
    return ", ".join(f"{name}={value:.4g}" for name, value in natural.items())


def _box(bounds: Bounds, k: int) -> list[tuple[float, float]]:
    """Normalise ``bounds`` into a per-factor list of ``(low, high)`` pairs."""
    if (
        len(bounds) == 2
        and np.isscalar(bounds[0])
        and np.isscalar(bounds[1])
    ):
        lo, hi = float(bounds[0]), float(bounds[1])  # type: ignore[arg-type]
        return [(lo, hi)] * k
    pairs = [(float(lo), float(hi)) for lo, hi in bounds]  # type: ignore[misc]
    if len(pairs) != k:
        raise ValueError(f"expected {k} bound pairs, got {len(pairs)}")
    return pairs


# --------------------------------------------------------------------------- #
# Stationary point + canonical analysis
# --------------------------------------------------------------------------- #


@dataclass
class StationaryPoint:
    """The unconstrained stationary point of a fitted quadratic surface.

    ``kind`` comes from the canonical analysis (the signs of the eigenvalues of ``B``):
    all negative -> ``"maximum"``, all positive -> ``"minimum"``, mixed -> ``"saddle"``.
    """

    coded: np.ndarray
    natural: dict[str, float]
    response: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    kind: Literal["maximum", "minimum", "saddle"]

    def __repr__(self) -> str:
        return (
            f"StationaryPoint({self.kind}: {_format_point(self.natural)} "
            f"-> {self.response:.4g})"
        )


def stationary_point(result: FitResult) -> StationaryPoint:
    """Stationary point ``x_s = -1/2 B^-1 b`` of a fitted second-order model.

    Requires the model to have curvature (a non-zero ``B``); a purely linear model has no
    stationary point and raises ``ValueError``.
    """
    b0, b, big_b = _quadratic_form(result)
    # tolerance tied to the coefficient scale so floating-point interaction/round-off noise
    # (~1e-16 against O(1) effects) reads as no curvature rather than a spurious full rank.
    scale = max(1.0, float(np.abs(result.coefficients).max()))
    if np.linalg.matrix_rank(big_b, tol=1e-8 * scale) < big_b.shape[0]:
        raise ValueError(
            "model curvature is rank-deficient (no unique stationary point); "
            "fit a quadratic model or use optimum() for a constrained search"
        )

    # The stationary point is where the fitted surface is flat (zero gradient): the candidate
    # optimum. Gradient is b + 2 B x; setting it to zero gives 2 B x = -b.
    x_s = np.linalg.solve(2.0 * big_b, -b)
    # Canonical analysis: the eigenvalues of B describe the surface's curvature along its
    # principal axes. All negative -> the surface curves down in every direction (a true
    # maximum); all positive -> a minimum; mixed signs -> a saddle (best on the boundary, where
    # optimum() should be used instead). Eigenvectors give the directions of steepest curvature.
    eigenvalues, eigenvectors = np.linalg.eigh(big_b)

    tol = 1e-9 * max(1.0, float(np.abs(eigenvalues).max()))
    if np.all(eigenvalues < -tol):
        kind: Literal["maximum", "minimum", "saddle"] = "maximum"
    elif np.all(eigenvalues > tol):
        kind = "minimum"
    else:
        kind = "saddle"

    return StationaryPoint(
        coded=x_s,
        natural=_decode(result.factors, x_s),
        response=_predict(b0, b, big_b, x_s),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kind=kind,
    )


# --------------------------------------------------------------------------- #
# Constrained optimum over the coded box
# --------------------------------------------------------------------------- #


@dataclass
class Optimum:
    """The constrained optimum of a fitted surface over the coded design box."""

    coded: np.ndarray
    natural: dict[str, float]
    response: float
    maximize: bool
    at_bound: bool

    def __repr__(self) -> str:
        direction = "max" if self.maximize else "min"
        bound = " (at bound)" if self.at_bound else ""
        return (
            f"Optimum({direction}: {_format_point(self.natural)} "
            f"-> {self.response:.4g}{bound})"
        )


def _multistart_minimize(
    objective: Callable[[np.ndarray], float],
    box: list[tuple[float, float]],
    *,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Minimise ``objective`` over ``box`` from several starts; return the best point."""
    lows = np.array([lo for lo, _ in box])
    highs = np.array([hi for _, hi in box])
    rng = np.random.default_rng(0)
    # A quadratic constrained to a box can have its optimum in the interior or on any face/
    # corner; a single gradient descent can stall at a local optimum or a saddle. Starting from
    # the center, both bounds and several random points and keeping the best makes the search
    # robust to that.
    starts = [
        0.5 * (lows + highs),  # center
        lows.copy(),
        highs.copy(),
    ]
    starts.extend(rng.uniform(lows, highs) for _ in range(8 * len(box)))

    best_x: np.ndarray | None = None
    best_f = np.inf
    for x0 in starts:
        res = sciopt.minimize(objective, x0, jac=jac, bounds=box, method="L-BFGS-B")
        if res.fun < best_f:
            best_f = float(res.fun)
            best_x = np.clip(res.x, lows, highs)
    assert best_x is not None
    return best_x


def optimum(
    result: FitResult,
    *,
    maximize: bool = True,
    bounds: Bounds = (-1.0, 1.0),
) -> Optimum:
    """Constrained optimum of the fitted surface over the coded box.

    ``bounds`` is either a single ``(low, high)`` applied to every factor (default: the
    coded design region ``[-1, 1]``) or a per-factor sequence of ``(low, high)`` pairs.
    Use this when :func:`stationary_point` falls outside the feasible region.
    """
    b0, b, big_b = _quadratic_form(result)
    box = _box(bounds, len(result.factors))
    sign = -1.0 if maximize else 1.0

    def objective(x: np.ndarray) -> float:
        return sign * _predict(b0, b, big_b, x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return np.asarray(sign * (b + 2.0 * big_b @ x), dtype=float)

    x_opt = _multistart_minimize(objective, box, jac=gradient)

    lows = np.array([lo for lo, _ in box])
    highs = np.array([hi for _, hi in box])
    at_bound = bool(
        np.any(np.isclose(x_opt, lows, atol=1e-6)) or np.any(np.isclose(x_opt, highs, atol=1e-6))
    )
    return Optimum(
        coded=x_opt,
        natural=_decode(result.factors, x_opt),
        response=_predict(b0, b, big_b, x_opt),
        maximize=maximize,
        at_bound=at_bound,
    )


# --------------------------------------------------------------------------- #
# Multi-response desirability (Derringer-Suich)
# --------------------------------------------------------------------------- #


@dataclass
class ResponseGoal:
    """A desirability specification for one fitted response.

    ``goal`` is ``"max"``, ``"min"`` or ``"target"``. ``low``/``high`` bracket the range
    over which desirability ramps from 0 to 1; ``target`` is required for ``"target"``.
    ``weight`` shapes the ramp (>1 emphasises getting close to the ideal, <1 relaxes it).
    """

    result: FitResult
    goal: Goal
    low: float
    high: float
    target: float | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("high must exceed low")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if self.goal == "target":
            if self.target is None:
                raise ValueError("target goal requires a target value")
            if not self.low < self.target < self.high:
                raise ValueError("target must lie strictly between low and high")

    def desirability(self, value: float) -> float:
        """Map a predicted response ``value`` to a desirability in ``[0, 1]``.

        Desirability puts responses on different scales onto a common 0 (unacceptable) to 1
        (ideal) ruler. "max" ramps up across ``[low, high]``, "min" ramps down, and "target"
        peaks at the target and falls off either side. The ``weight`` exponent bends the ramp:
        ``>1`` demands getting near the ideal before desirability rises, ``<1`` is lenient.
        """
        lo, hi, wt = self.low, self.high, self.weight
        if self.goal == "max":
            if value <= lo:
                return 0.0
            if value >= hi:
                return 1.0
            return float(((value - lo) / (hi - lo)) ** wt)
        if self.goal == "min":
            if value <= lo:
                return 1.0
            if value >= hi:
                return 0.0
            return float(((hi - value) / (hi - lo)) ** wt)
        # target
        assert self.target is not None
        t = self.target
        if value < lo or value > hi:
            return 0.0
        if value <= t:
            return float(((value - lo) / (t - lo)) ** wt)
        return float(((hi - value) / (hi - t)) ** wt)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the goal *definition* for storage.

        The bound :class:`FitResult` is deliberately omitted: like other analysis results it
        is *derived* from a design plus a response and is re-fit on load, so only the goal
        specification (direction, ramp bounds, target, weight) is data worth persisting.
        Pair this with :meth:`from_dict`, which takes the re-fitted result back in.
        """
        return {
            "goal": self.goal,
            "low": self.low,
            "high": self.high,
            "target": self.target,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], result: FitResult) -> ResponseGoal:
        """Rebuild a :class:`ResponseGoal` from :meth:`to_dict` plus a re-fitted ``result``."""
        goal = data["goal"]
        if goal not in ("max", "min", "target"):
            raise ValueError(f"unknown goal {goal!r}; expected 'max', 'min' or 'target'")
        target = data.get("target")
        return cls(
            result=result,
            goal=goal,
            low=float(data["low"]),
            high=float(data["high"]),
            target=None if target is None else float(target),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class DesirabilityResult:
    """The point maximising overall Derringer-Suich desirability over the coded box."""

    coded: np.ndarray
    natural: dict[str, float]
    responses: np.ndarray
    individual: np.ndarray
    overall: float

    def __repr__(self) -> str:
        responses = ", ".join(f"{v:.4g}" for v in self.responses)
        return (
            f"DesirabilityResult(D={self.overall:.4g}: {_format_point(self.natural)} "
            f"| responses=[{responses}])"
        )


def desirability(
    goals: Sequence[ResponseGoal], *, bounds: Bounds = (-1.0, 1.0)
) -> DesirabilityResult:
    """Maximise the geometric-mean desirability of several responses over the coded box.

    Each response is predicted from its own :class:`FitResult` (all must share the same
    factors). The overall desirability is ``D = (prod d_i)^(1/m)``, maximised with a global
    optimiser because the per-response desirabilities are non-smooth (flat where saturated).
    """
    if not goals:
        raise ValueError("need at least one response goal")

    # Every response is optimized over one shared coded box, so the goals must agree not just on
    # factor *names* but on each factor's definition (bounds/levels) -- otherwise a coded point
    # would decode to different natural settings per response. Frozen factor dataclasses compare
    # by value, so tuple equality checks names, order, and coding together.
    reference = tuple(goals[0].result.factors)
    names = goals[0].result.factors.names
    for goal in goals[1:]:
        if tuple(goal.result.factors) != reference:
            raise ValueError("all responses must be fitted over the same factors")

    forms = [_quadratic_form(goal.result) for goal in goals]
    box = _box(bounds, len(names))
    m = len(goals)

    def neg_overall(x: np.ndarray) -> float:
        d = np.empty(m)
        for i, (goal, (b0, b, big_b)) in enumerate(zip(goals, forms, strict=True)):
            d[i] = goal.desirability(_predict(b0, b, big_b, x))
        # The overall score is the *geometric* mean, so any single unacceptable response (d=0)
        # drags the whole product to zero -- a deliberate veto: a compromise must satisfy every
        # goal at least minimally, not let a great score on one paper over a failure on another.
        if np.any(d <= 0.0):
            return 0.0  # geometric mean is 0; negate -> 0
        return -float(np.prod(d) ** (1.0 / m))

    res = sciopt.differential_evolution(neg_overall, box, rng=0, tol=1e-10, polish=True)
    x_opt = np.clip(res.x, [lo for lo, _ in box], [hi for _, hi in box])

    responses = np.array([_predict(*form, x_opt) for form in forms])
    individual = np.array([goal.desirability(v) for goal, v in zip(goals, responses, strict=True)])
    overall = float(-neg_overall(x_opt))
    return DesirabilityResult(
        coded=x_opt,
        natural=_decode(goals[0].result.factors, x_opt),
        responses=responses,
        individual=individual,
        overall=overall,
    )
