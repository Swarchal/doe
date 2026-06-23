import numpy as np
import pytest

from doe.factors import CategoricalFactor, ContinuousFactor, FactorSet


def test_continuous_coding_roundtrip():
    f = ContinuousFactor("temp", low=20.0, high=80.0, units="C")
    natural = np.array([20.0, 50.0, 80.0])
    coded = f.code(natural)
    assert np.allclose(coded, [-1.0, 0.0, 1.0])
    assert np.allclose(f.decode(coded), natural)


def test_continuous_rejects_bad_bounds():
    with pytest.raises(ValueError):
        ContinuousFactor("x", low=5.0, high=5.0)


def test_categorical_needs_two_levels():
    with pytest.raises(ValueError):
        CategoricalFactor("catalyst", levels=("A",))


def test_factorset_unique_names_and_lookup():
    a = ContinuousFactor("a", 0, 1)
    b = ContinuousFactor("b", 0, 1)
    fs = FactorSet([a, b])
    assert fs.names == ["a", "b"]
    assert fs["b"] is b
    with pytest.raises(ValueError):
        FactorSet([a, ContinuousFactor("a", 0, 2)])
