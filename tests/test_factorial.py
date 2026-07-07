import numpy as np
import pytest

from doe.analysis.fit import fit_ols
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.factorial import (
    fractional_factorial,
    full_factorial,
    plackett_burman,
)


def _factors(n):
    return [ContinuousFactor(chr(ord("a") + i), low=0.0, high=10.0) for i in range(n)]


def test_full_factorial_run_count():
    design = full_factorial(_factors(3), levels=2)
    assert design.n_runs == 8
    assert design.factors.names == ["a", "b", "c"]


def test_full_factorial_corner_values_in_natural_units():
    design = full_factorial(_factors(2), levels=2)
    # corners of a 0..10 box
    assert set(map(tuple, design.runs.to_numpy())) == {
        (0.0, 0.0),
        (0.0, 10.0),
        (10.0, 0.0),
        (10.0, 10.0),
    }


def test_full_factorial_coded_is_plus_minus_one():
    design = full_factorial(_factors(2), levels=2)
    coded = design.coded().to_numpy()
    assert set(np.unique(coded)) == {-1.0, 1.0}


def test_fractional_factorial_half_fraction():
    # 2^(4-1) with D = ABC -> 8 runs
    design = fractional_factorial(_factors(4), generators=["D=ABC"])
    assert design.n_runs == 8
    coded = design.coded().to_numpy()
    # the defining relation D = A*B*C must hold for every run
    assert np.allclose(coded[:, 3], coded[:, 0] * coded[:, 1] * coded[:, 2])


def test_fractional_factorial_uses_generator_lhs_for_column_order():
    design = fractional_factorial(_factors(5), generators=["E=ABC", "D=AB"])
    coded = design.coded().to_numpy()

    assert np.allclose(coded[:, 3], coded[:, 0] * coded[:, 1])
    assert np.allclose(coded[:, 4], coded[:, 0] * coded[:, 1] * coded[:, 2])


def test_fractional_factorial_rejects_unknown_generator_lhs():
    with pytest.raises(ValueError, match="left-hand side"):
        fractional_factorial(_factors(4), generators=["Z=ABC"])


def test_fractional_factorial_rejects_categorical_with_more_than_two_levels():
    factors = [
        CategoricalFactor("mode", ("A", "B", "C")),
        *_factors(2),
    ]
    with pytest.raises(ValueError, match="two-level designs cannot encode"):
        fractional_factorial(factors, generators=["c=AB"])


def test_fractional_factorial_accepts_two_level_categorical():
    factors = [
        CategoricalFactor("mode", ("low", "high")),
        *_factors(2),
    ]
    design = fractional_factorial(factors, generators=["b=AB"])
    assert sorted(design.runs["mode"].unique()) == ["high", "low"]


def test_plackett_burman_run_count():
    # N is the smallest constructible multiple of 4 with N >= k + 1
    assert plackett_burman(_factors(3)).n_runs == 4
    assert plackett_burman(_factors(7)).n_runs == 8
    assert plackett_burman(_factors(8)).n_runs == 12  # needs N >= 9 -> 12
    assert plackett_burman(_factors(11)).n_runs == 12
    assert plackett_burman(_factors(19)).n_runs == 20


def test_plackett_burman_factor_columns_match_factor_count():
    design = plackett_burman(_factors(5))
    assert design.factors.names == ["a", "b", "c", "d", "e"]
    assert design.coded().shape == (8, 5)


def test_plackett_burman_coded_is_plus_minus_one():
    coded = plackett_burman(_factors(5)).coded().to_numpy()
    assert set(np.unique(coded)) == {-1.0, 1.0}


def test_plackett_burman_rejects_categorical_with_more_than_two_levels():
    factors = [
        CategoricalFactor("mode", ("A", "B", "C")),
        *_factors(7),
    ]
    with pytest.raises(ValueError, match="two-level designs cannot encode"):
        plackett_burman(factors)


def test_plackett_burman_accepts_two_level_categorical():
    factors = [
        CategoricalFactor("mode", ("low", "high")),
        *_factors(3),
    ]
    design = plackett_burman(factors)
    assert sorted(design.runs["mode"].unique()) == ["high", "low"]


def test_plackett_burman_natural_units_are_corners():
    design = plackett_burman(_factors(7))
    assert set(np.unique(design.runs.to_numpy())) == {0.0, 10.0}


def test_plackett_burman_columns_balanced_and_orthogonal():
    # the classic 12-run design (11 factors) is the canonical correctness anchor
    coded = plackett_burman(_factors(11)).coded().to_numpy()
    n = coded.shape[0]
    assert coded.shape == (12, 11)
    # every factor column is balanced (equal numbers of +1 and -1)
    assert np.allclose(coded.sum(axis=0), 0.0)
    # columns are mutually orthogonal: X^T X = n * I
    assert np.allclose(coded.T @ coded, n * np.eye(coded.shape[1]))


def test_plackett_burman_12_run_is_not_a_regular_fraction():
    # PB-12 famously has *partial* (fractional) aliasing: no two- or three-factor
    # product collapses onto a single column the way a regular generator would.
    coded = plackett_burman(_factors(11)).coded().to_numpy()
    ncols = coded.shape[1]
    for i in range(ncols):
        for j in range(i + 1, ncols):
            prod = coded[:, i] * coded[:, j]
            assert not any(
                np.allclose(prod, coded[:, k]) or np.allclose(prod, -coded[:, k])
                for k in range(ncols)
            )


def test_plackett_burman_recovers_main_effects():
    # main effects are estimable and recovered exactly from a noiseless response
    design = plackett_burman(_factors(7))
    coded = design.coded().to_numpy()
    # y = 5 + 2*a - 3*c + 1*g  (in coded units)
    y = 5 + 2 * coded[:, 0] - 3 * coded[:, 2] + 1 * coded[:, 6]

    result = fit_ols(design, y, order=1, interactions=False)
    summary = result.summary()

    assert np.isclose(summary["Intercept"][0], 5.0)
    # effect = 2 * coefficient in coded units
    assert np.isclose(summary["a"][1], 4.0)
    assert np.isclose(summary["c"][1], -6.0)
    assert np.isclose(summary["g"][1], 2.0)
    assert np.isclose(summary["b"][1], 0.0, atol=1e-9)


def test_plackett_burman_rejects_unconstructible_size():
    # 27 factors would need N = 28, which this construction does not provide;
    # the next constructible size (32, a power of two) is used instead.
    assert plackett_burman(_factors(27)).n_runs == 32


# --------------------------------------------------------------------------- #
# Generator spec in meta (the serialized design-spec layer)
# --------------------------------------------------------------------------- #


def test_full_factorial_records_generator_spec():
    design = full_factorial(_factors(2), levels=[2, 3])
    assert design.meta["generator"] == {
        "name": "full_factorial",
        "parameters": {"levels": [2, 3]},
    }


def test_fractional_factorial_spec_regenerates_design():
    # the defining relation is unrecoverable from the run table alone; recording it in
    # meta is what lets a serialized design reconstruct the intended experiment.
    design = fractional_factorial(_factors(4), ["D=ABC"])
    spec = design.meta["generator"]
    assert spec["name"] == "fractional_factorial"
    assert spec["parameters"] == {"generators": ["D=ABC"]}
    rebuilt = fractional_factorial(_factors(4), **spec["parameters"])
    assert rebuilt.runs.equals(design.runs)


def test_plackett_burman_records_generator_spec():
    design = plackett_burman(_factors(7))
    assert design.meta["generator"] == {"name": "plackett_burman", "parameters": {}}
