import numpy as np
import pytest

from doe.analysis.model import build_model_matrix, coded_design_points, expand_coded_points
from doe.factors import CategoricalFactor, ContinuousFactor, FactorSet
from doe.generators.factorial import full_factorial
from doe.generators.optimal import candidate_grid


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


# --------------------------------------------------------------------------- #
# Array-based coded-point expansion for Phase 3 diagnostics / optimal designs
# --------------------------------------------------------------------------- #


def test_expand_coded_points_matches_linear_design_matrix():
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    design = full_factorial(factors)

    expected = build_model_matrix(design, order=1, interactions=True)
    points = design.coded()[design.factors.names].to_numpy()
    got = expand_coded_points(points, design.factors, order=1, interactions=True)

    assert got.term_names == expected.term_names
    assert np.allclose(got.X, expected.X)


def test_expand_coded_points_matches_quadratic_design_matrix():
    factors = [ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)]
    design = full_factorial(factors, levels=3)

    expected = build_model_matrix(design, order=2, interactions=True)
    points = design.coded()[design.factors.names].to_numpy()
    got = expand_coded_points(points, design.factors, order=2, interactions=True)

    assert got.term_names == expected.term_names
    assert np.allclose(got.X, expected.X)


def test_expand_coded_points_require_squares_forces_pure_pm1_squared_column():
    # a pure +/-1 point can't trigger the off-+/-1 heuristic; require_squares overrides it
    # for the named factors, which is how FitResult.predict scores single corner points.
    factors = FactorSet([ContinuousFactor("a", 0.0, 10.0), ContinuousFactor("b", 0.0, 10.0)])
    points = np.array([[-1.0, -1.0], [1.0, 1.0]])

    default = expand_coded_points(points, factors, order=2, interactions=True)
    assert "a^2" not in default.term_names
    assert "b^2" not in default.term_names

    forced = expand_coded_points(
        points, factors, order=2, interactions=True, require_squares=["a", "b"]
    )
    assert forced.term_names[-2:] == ["a^2", "b^2"]
    assert np.allclose(forced.as_frame()["a^2"].to_numpy(), 1.0)
    assert np.allclose(forced.as_frame()["b^2"].to_numpy(), 1.0)

    # a factor that already takes off-+/-1 values only gets its square emitted once
    mixed_points = np.array([[0.0, -1.0], [1.0, 1.0]])
    once = expand_coded_points(
        mixed_points, factors, order=2, interactions=True, require_squares=["a"]
    )
    assert once.term_names.count("a^2") == 1


def test_expand_coded_points_matches_mixed_design_matrix():
    factors = [
        ContinuousFactor("temp", 0.0, 100.0),
        CategoricalFactor("catalyst", ("A", "B", "C")),
    ]
    design = full_factorial(factors, levels=3)

    expected = build_model_matrix(design, order=2, interactions=True)
    points = candidate_grid(factors, levels=3)
    got = expand_coded_points(points, design.factors, order=2, interactions=True)

    assert got.term_names == expected.term_names
    assert np.allclose(got.X, expected.X)


def test_coded_design_points_maps_categorical_labels_to_candidate_coordinates():
    factors = [
        ContinuousFactor("temp", 0.0, 100.0),
        CategoricalFactor("catalyst", ("A", "B", "C")),
    ]
    design = full_factorial(factors, levels=3)

    points = coded_design_points(design)

    assert points.shape == (9, 2)
    assert set(np.unique(points[:, 0])) == {-1.0, 0.0, 1.0}
    assert set(np.unique(points[:, 1])) == {-1.0, 0.0, 1.0}
    assert np.all(points[design.runs["catalyst"] == "A", 1] == -1.0)
    assert np.all(points[design.runs["catalyst"] == "B", 1] == 0.0)
    assert np.all(points[design.runs["catalyst"] == "C", 1] == 1.0)


def test_coded_design_points_rejects_unknown_categorical_level():
    design = full_factorial([CategoricalFactor("c", ("A", "B"))])
    design.runs.loc[0, "c"] = "Z"
    with pytest.raises(ValueError, match="unknown level"):
        coded_design_points(design)


def test_expand_coded_points_rejects_invalid_categorical_coordinate():
    design = full_factorial([CategoricalFactor("c", ("A", "B", "C"))])
    with pytest.raises(ValueError, match="discrete coded levels"):
        expand_coded_points(np.array([[-0.5], [1.0]]), design.factors)
