import numpy as np

from doe.analysis.fit import fit_ols
from doe.factors import ContinuousFactor
from doe.generators.factorial import full_factorial


def test_fit_recovers_known_effects():
    factors = [ContinuousFactor("a", 0, 10), ContinuousFactor("b", 0, 10)]
    design = full_factorial(factors, levels=2)
    coded = design.coded().to_numpy()

    # response built in coded units: y = 10 + 3*a + 2*b + 1.5*a*b
    a, b = coded[:, 0], coded[:, 1]
    y = 10 + 3 * a + 2 * b + 1.5 * a * b

    result = fit_ols(design, y, order=1, interactions=True)
    summary = result.summary()

    assert np.isclose(summary["Intercept"][0], 10.0)
    # effect = 2 * coefficient in coded units
    assert np.isclose(summary["a"][1], 6.0)
    assert np.isclose(summary["b"][1], 4.0)
    assert np.isclose(summary["a:b"][1], 3.0)
    assert np.isclose(result.r_squared, 1.0)


def test_fit_response_length_mismatch():
    design = full_factorial([ContinuousFactor("a", 0, 1)], levels=2)
    try:
        fit_ols(design, np.array([1.0, 2.0, 3.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched response length")
