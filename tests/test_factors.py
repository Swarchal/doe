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


# --- hard_to_change flag + whole-plot partitions (Phase 5b) ---------------------------------


def test_hard_to_change_defaults_false_and_omitted_from_dict():
    f = ContinuousFactor("temp", low=20.0, high=80.0)
    assert f.hard_to_change is False
    assert "hard_to_change" not in f.to_dict()  # byte-stable with pre-5b documents


def test_hard_to_change_round_trips_when_set():
    f = ContinuousFactor("oven", low=200, high=400, hard_to_change=True)
    assert f.to_dict()["hard_to_change"] is True
    assert ContinuousFactor.from_dict(f.to_dict()).hard_to_change is True
    c = CategoricalFactor("lot", ("A", "B"), hard_to_change=True)
    assert c.to_dict()["hard_to_change"] is True
    assert CategoricalFactor.from_dict(c.to_dict()).hard_to_change is True


def test_factorset_whole_plot_partitions():
    oven = ContinuousFactor("oven", 200, 400, hard_to_change=True)
    time = ContinuousFactor("time", 5, 15)
    cat = CategoricalFactor("mode", ("x", "y"))
    fs = FactorSet([oven, time, cat])
    assert [f.name for f in fs.whole_plot_factors] == ["oven"]
    assert [f.name for f in fs.sub_plot_factors] == ["time", "mode"]
