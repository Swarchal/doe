"""Anchors for the space-filling generators and coverage diagnostics.

See the design docs for the anchor rationale.
"""

import numpy as np
import pytest

import doe
from doe import (
    CategoricalFactor,
    ContinuousFactor,
    discrepancy,
    full_factorial,
    halton,
    latin_hypercube,
    maximin_distance,
    sobol,
)


@pytest.fixture
def factors() -> list[ContinuousFactor]:
    return [
        ContinuousFactor("temperature", low=20.0, high=80.0, units="C"),
        ContinuousFactor("time", low=1.0, high=5.0, units="h"),
        ContinuousFactor("concentration", low=0.1, high=0.9, units="M"),
    ]


def _stratum_indices(coded_column: np.ndarray, n_runs: int) -> np.ndarray:
    """Map coded [-1, +1] values to their 0-based stratum index out of ``n_runs``."""
    unit = (np.asarray(coded_column, dtype=float) + 1.0) / 2.0
    return np.clip(np.floor(unit * n_runs), 0, n_runs - 1).astype(int)


def test_spacefilling_public_api_exports_are_available():
    for name in ["latin_hypercube", "sobol", "halton", "discrepancy", "maximin_distance"]:
        assert name in doe.__all__
        assert getattr(doe, name) is not None


def test_lhs_places_exactly_one_point_per_stratum_per_factor(factors):
    n_runs = 12
    design = latin_hypercube(factors, n_runs=n_runs, seed=0)
    coded = design.coded()
    assert design.n_runs == n_runs
    for name in coded.columns:
        strata = _stratum_indices(coded[name].to_numpy(), n_runs)
        assert sorted(strata) == list(range(n_runs))


def test_lhs_is_reproducible_from_seed_and_self_describing(factors):
    a = latin_hypercube(factors, n_runs=10, seed=42)
    b = latin_hypercube(factors, n_runs=10, seed=42)
    assert a.runs.equals(b.runs)
    assert a.meta["sampler"] == "lhs"
    assert a.meta["seed"] == 42
    assert a.meta["criterion"] == "maximin"


def test_sobol_requires_power_of_two_run_counts(factors):
    design = sobol(factors, n_runs=8, seed=0)
    assert design.n_runs == 8
    assert design.meta["sampler"] == "sobol"
    with pytest.raises(ValueError, match=r"8.*16"):
        sobol(factors, n_runs=10, seed=0)


def test_scrambled_sobol_beats_iid_uniform_on_discrepancy(factors):
    design = sobol(factors, n_runs=16, scramble=True, seed=0)
    rng = np.random.default_rng(0)
    iid = latin_hypercube(factors, n_runs=16, criterion=None, seed=0)
    # compare against a genuinely i.i.d. cloud, not the stratified LHS
    iid_runs = iid.runs.copy()
    for factor in factors:
        iid_runs[factor.name] = factor.decode(rng.uniform(-1.0, 1.0, size=16))
    iid = doe.Design(iid_runs, iid.factors)
    assert discrepancy(design) < discrepancy(iid)


def test_halton_accepts_arbitrary_run_counts(factors):
    design = halton(factors, n_runs=10, seed=0)
    assert design.n_runs == 10
    assert design.meta["sampler"] == "halton"


def test_maximin_distance_anchors():
    factors = [ContinuousFactor("A", -1.0, 1.0), ContinuousFactor("B", -1.0, 1.0)]
    square = full_factorial(factors)
    # the 2^2 corners in the unit cube are the cube's corners: min pairwise distance = side = 1
    assert maximin_distance(square) == pytest.approx(1.0)
    duplicated = square.replicate(2)
    assert maximin_distance(duplicated) == pytest.approx(0.0)


def test_categorical_factors_are_rejected(factors):
    mixed = [*factors, CategoricalFactor("catalyst", levels=("Pd", "Pt"))]
    for generator in (latin_hypercube, sobol, halton):
        with pytest.raises(ValueError, match="continuous"):
            generator(mixed, n_runs=8)
