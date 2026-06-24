import numpy as np
import pytest

from doe.analysis.model import build_model_matrix
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.factorial import full_factorial


def test_quadratic_drops_squares_for_two_level_factors():
    # a pure +/-1 factor has x^2 == 1 (collinear with the intercept), so a quadratic
    # model must NOT emit a squared column for it -- otherwise X is rank-deficient.
    design = full_factorial([ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)])
    mm = build_model_matrix(design, order=2, interactions=True)
    assert not any(name.endswith("^2") for name in mm.term_names)


def test_quadratic_keeps_squares_for_three_level_factors():
    # a 3-level factor genuinely takes a value off {-1, +1}, so its square is estimable.
    design = full_factorial([ContinuousFactor("a", 0.0, 10.0)], levels=3)
    mm = build_model_matrix(design, order=2, interactions=True)
    assert "a^2" in mm.term_names


# --------------------------------------------------------------------------- #
# Categorical contrast (effect/deviation) coding
# --------------------------------------------------------------------------- #


def test_categorical_two_level_is_pm1_effect_contrast():
    # a 2-level categorical reduces to a single +/-1 column matching the generator's
    # corner coding: the first level is the -1 reference, the second is +1.
    design = full_factorial([CategoricalFactor("catalyst", ("A", "B"))])
    mm = build_model_matrix(design, interactions=False)
    assert mm.term_names == ["Intercept", "catalyst[B]"]

    coded = design.coded()["catalyst"].to_numpy()
    col = mm.X[:, mm.term_names.index("catalyst[B]")]
    assert np.all(col[coded == "B"] == 1.0)
    assert np.all(col[coded == "A"] == -1.0)
    # balanced -> orthogonal to the intercept
    assert np.isclose(col.sum(), 0.0)


def test_categorical_three_level_makes_k_minus_1_contrasts():
    design = full_factorial([CategoricalFactor("solvent", ("X", "Y", "Z"))])
    mm = build_model_matrix(design, interactions=False)
    assert mm.term_names == ["Intercept", "solvent[Y]", "solvent[Z]"]
    assert mm.X.shape == (3, 3)

    coded = design.coded()["solvent"].to_numpy()
    col_y = mm.X[:, mm.term_names.index("solvent[Y]")]
    # effect coding: +1 for that level, -1 for the reference (X), 0 elsewhere
    assert np.all(col_y[coded == "Y"] == 1.0)
    assert np.all(col_y[coded == "X"] == -1.0)
    assert np.all(col_y[coded == "Z"] == 0.0)


def test_continuous_categorical_interaction_is_product_of_columns():
    temp = ContinuousFactor("temp", 0.0, 100.0)
    catalyst = CategoricalFactor("catalyst", ("A", "B"))
    design = full_factorial([temp, catalyst])
    mm = build_model_matrix(design, order=1, interactions=True)

    assert "temp:catalyst[B]" in mm.term_names
    inter = mm.X[:, mm.term_names.index("temp:catalyst[B]")]
    temp_col = mm.X[:, mm.term_names.index("temp")]
    cat_col = mm.X[:, mm.term_names.index("catalyst[B]")]
    assert np.allclose(inter, temp_col * cat_col)


def test_categorical_categorical_interaction_count():
    # a (3-level -> 2 contrasts) x b (2-level -> 1 contrast) gives 2*1 interaction columns
    a = CategoricalFactor("a", ("a1", "a2", "a3"))
    b = CategoricalFactor("b", ("b1", "b2"))
    design = full_factorial([a, b])
    mm = build_model_matrix(design, interactions=True)
    interactions = [name for name in mm.term_names if ":" in name]
    assert len(interactions) == 2 * 1
    assert set(interactions) == {"a[a2]:b[b2]", "a[a3]:b[b2]"}


def test_quadratic_never_squares_categorical_columns():
    a = ContinuousFactor("a", 0.0, 10.0)
    cat = CategoricalFactor("c", ("x", "y", "z"))
    design = full_factorial([a, cat], levels=3)  # 'a' takes 3 levels -> a^2 is estimable
    mm = build_model_matrix(design, order=2, interactions=False)
    assert "a^2" in mm.term_names
    assert not any("^2" in name and "c[" in name for name in mm.term_names)


def test_categorical_rejects_unknown_level():
    design = full_factorial([CategoricalFactor("c", ("A", "B"))])
    design.runs.loc[0, "c"] = "Z"
    with pytest.raises(ValueError, match="unknown level"):
        build_model_matrix(design)
