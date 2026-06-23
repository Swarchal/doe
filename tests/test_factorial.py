import numpy as np
import pytest

from doe.factors import ContinuousFactor
from doe.generators.factorial import fractional_factorial, full_factorial


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
