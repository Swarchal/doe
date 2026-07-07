"""Anchors for mixture designs and Scheffé blending analysis.

The published references are
Cornell's *Experiments with Mixtures* (yarn-elongation `{3, 2}` lattice fit) and the
McLean-Anderson (1966) railroad-flare extreme-vertices example.
"""

import numpy as np
import pandas as pd
import pytest

import doe
from doe import (
    ContinuousFactor,
    Design,
    FactorSet,
    MixtureFactor,
    anova_table,
    d_optimal,
    extreme_vertices,
    factor_from_dict,
    fit_ols,
    log_det_information,
    mixture_candidates,
    simplex_centroid,
    simplex_lattice,
    validate_design_dict,
)
from doe.analysis.model import build_model_matrix


@pytest.fixture
def components() -> list[MixtureFactor]:
    return [
        MixtureFactor("polyethylene"),
        MixtureFactor("polystyrene"),
        MixtureFactor("polypropylene"),
    ]


def _rows_as_set(design: Design) -> set[tuple[float, ...]]:
    return {tuple(np.round(row, 9)) for row in design.runs.to_numpy(dtype=float)}


def test_mixture_public_api_exports_are_available():
    for name in [
        "MixtureFactor",
        "simplex_lattice",
        "simplex_centroid",
        "extreme_vertices",
        "mixture_candidates",
    ]:
        assert name in doe.__all__
        assert getattr(doe, name) is not None


# --------------------------------------------------------------------------- #
# MixtureFactor + FactorSet rules
# --------------------------------------------------------------------------- #


def test_mixture_factor_validates_proportion_bounds():
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        MixtureFactor("a", low=-0.1, high=0.5)
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        MixtureFactor("a", low=0.5, high=0.2)
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        MixtureFactor("a", low=0.0, high=1.2)


def test_factor_set_rejects_mixing_mixture_with_other_types():
    with pytest.raises(ValueError, match="cannot be combined"):
        FactorSet([MixtureFactor("a"), ContinuousFactor("temp", low=0.0, high=1.0)])


def test_factor_set_rejects_infeasible_mixture_bounds():
    # sum(low) > 1: no blend can satisfy all lower bounds simultaneously
    with pytest.raises(ValueError, match="no feasible blend"):
        FactorSet([MixtureFactor("a", low=0.6), MixtureFactor("b", low=0.6)])


def test_is_mixture_property(components: list[MixtureFactor]):
    assert FactorSet(components).is_mixture
    assert not FactorSet([ContinuousFactor("x", low=0.0, high=1.0)]).is_mixture


def test_mixture_factor_round_trips_through_dict():
    factor = MixtureFactor("binder", low=0.03, high=0.08, units="fraction")
    restored = factor_from_dict(factor.to_dict())
    assert restored == factor


def test_mixture_design_serialization_round_trip(components: list[MixtureFactor]):
    design = simplex_centroid(components)
    payload = design.to_dict()
    validate_design_dict(payload)
    restored = Design.from_dict(payload)
    assert restored.factors.is_mixture
    pd.testing.assert_frame_equal(restored.runs, design.runs)
    assert restored.point_types == design.point_types


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #


def test_simplex_lattice_3_2_gives_the_six_textbook_points(
    components: list[MixtureFactor],
):
    design = simplex_lattice(components, degree=2)
    expected = {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    }
    assert _rows_as_set(design) == expected
    assert design.n_runs == 6


def test_simplex_lattice_3_3_has_ten_runs(components: list[MixtureFactor]):
    # C(k + m - 1, m) = C(5, 3) = 10
    assert simplex_lattice(components, degree=3).n_runs == 10


def test_simplex_centroid_k3_has_seven_runs(components: list[MixtureFactor]):
    design = simplex_centroid(components)
    assert design.n_runs == 7  # 2^3 - 1
    assert design.point_types is not None
    assert design.point_types.count("vertex") == 3
    assert design.point_types.count("edge-centroid") == 3
    assert design.point_types.count("centroid") == 1


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_every_mixture_design_row_sums_to_one(
    components: list[MixtureFactor], degree: int
):
    for design in [
        simplex_lattice(components, degree=degree),
        simplex_centroid(components),
    ]:
        sums = design.runs.to_numpy(dtype=float).sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-12)


def test_lattice_and_centroid_require_unconstrained_components():
    constrained = [MixtureFactor("a", low=0.2), MixtureFactor("b"), MixtureFactor("c")]
    with pytest.raises(ValueError, match="extreme_vertices"):
        simplex_lattice(constrained, degree=2)
    with pytest.raises(ValueError, match="extreme_vertices"):
        simplex_centroid(constrained)


def test_extreme_vertices_reproduces_mclean_anderson_flare_vertices():
    # McLean & Anderson (1966) railroad flare: magnesium, sodium nitrate,
    # strontium nitrate, binder -- the published 8 extreme vertices.
    factors = [
        MixtureFactor("magnesium", low=0.40, high=0.60),
        MixtureFactor("sodium_nitrate", low=0.10, high=0.50),
        MixtureFactor("strontium_nitrate", low=0.10, high=0.50),
        MixtureFactor("binder", low=0.03, high=0.08),
    ]
    design = extreme_vertices(factors, include_centroid=False)
    expected = {
        (0.40, 0.10, 0.47, 0.03),
        (0.40, 0.10, 0.42, 0.08),
        (0.60, 0.10, 0.27, 0.03),
        (0.60, 0.10, 0.22, 0.08),
        (0.40, 0.47, 0.10, 0.03),
        (0.40, 0.42, 0.10, 0.08),
        (0.60, 0.27, 0.10, 0.03),
        (0.60, 0.22, 0.10, 0.08),
    }
    assert _rows_as_set(design) == expected
    sums = design.runs.to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)


def test_extreme_vertices_appends_the_region_centroid():
    factors = [
        MixtureFactor("a", low=0.2, high=0.6),
        MixtureFactor("b", low=0.1, high=0.6),
        MixtureFactor("c", low=0.1, high=0.5),
    ]
    design = extreme_vertices(factors)
    assert design.point_types is not None
    assert design.point_types[-1] == "centroid"
    vertices = design.runs.to_numpy(dtype=float)[:-1]
    centroid = design.runs.to_numpy(dtype=float)[-1]
    assert np.allclose(centroid, vertices.mean(axis=0))


def test_infeasible_bounds_raise():
    with pytest.raises(ValueError, match="no feasible blend"):
        extreme_vertices([MixtureFactor("a", low=0.7), MixtureFactor("b", low=0.7)])


def test_mixture_candidates_feasible_and_sum_to_one(components: list[MixtureFactor]):
    candidates = mixture_candidates(components, resolution=5)
    assert candidates.ndim == 2 and candidates.shape[1] == 3
    assert np.allclose(candidates.sum(axis=1), 1.0, atol=1e-9)
    # constrained region: every candidate respects the bounds
    constrained = [
        MixtureFactor("a", low=0.2, high=0.6),
        MixtureFactor("b", low=0.1, high=0.6),
        MixtureFactor("c", low=0.1, high=0.5),
    ]
    candidates = mixture_candidates(constrained, resolution=10)
    assert np.all(candidates >= np.array([0.2, 0.1, 0.1]) - 1e-9)
    assert np.all(candidates <= np.array([0.6, 0.6, 0.5]) + 1e-9)


# --------------------------------------------------------------------------- #
# Scheffé models
# --------------------------------------------------------------------------- #


def _yarn_design_and_response(
    components: list[MixtureFactor],
) -> tuple[Design, np.ndarray]:
    """Cornell's yarn-elongation data on the {3, 2} simplex lattice (replicated)."""
    rows = [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.0, 0.5, 0.5),
        (0.0, 0.5, 0.5),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.0, 0.5),
    ]
    y = np.array(
        [11.0, 12.4, 15.0, 14.8, 16.1, 8.8, 10.0, 10.0, 9.7, 11.8, 16.8, 16.0, 17.7, 16.4, 16.6]
    )
    fs = FactorSet(components)
    runs = pd.DataFrame(rows, columns=fs.names)
    return Design(runs, fs, name="yarn"), y


def test_scheffe_model_matrix_has_no_intercept_and_six_terms(
    components: list[MixtureFactor],
):
    design = simplex_lattice(components, degree=2)
    mm = build_model_matrix(design, order=2)
    assert "Intercept" not in mm.term_names
    assert len(mm.term_names) == 6  # 3 linear + C(3, 2) cross products
    assert mm.term_names[:3] == [f.name for f in components]
    assert all(":" in name for name in mm.term_names[3:])


def test_scheffe_quadratic_recovers_cornell_yarn_coefficients(
    components: list[MixtureFactor],
):
    design, y = _yarn_design_and_response(components)
    result = fit_ols(design, y, model="scheffe-quadratic")
    published = {
        "polyethylene": 11.7,
        "polystyrene": 9.4,
        "polypropylene": 16.4,
        "polyethylene:polystyrene": 19.0,
        "polyethylene:polypropylene": 11.4,
        "polystyrene:polypropylene": -9.6,
    }
    fitted = dict(zip(result.term_names, result.coefficients, strict=True))
    for term, value in published.items():
        assert fitted[term] == pytest.approx(value, abs=0.05)


def test_scheffe_fit_reports_centered_r2_and_nan_effects(
    components: list[MixtureFactor],
):
    design, y = _yarn_design_and_response(components)
    result = fit_ols(design, y, model="scheffe-quadratic")
    # the constant is in the Scheffé column space, so residuals sum to zero and the
    # centered R^2 is well-defined (and not the inflated uncorrected version)
    assert result.residuals.sum() == pytest.approx(0.0, abs=1e-9)
    ss_res = float(result.residuals @ result.residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    assert result.r_squared == pytest.approx(1.0 - ss_res / ss_tot)
    assert 0.9 < result.r_squared < 1.0
    assert np.all(np.isnan(result.effects))


def test_scheffe_model_names_require_a_mixture_design():
    design = doe.full_factorial([ContinuousFactor("x", low=0.0, high=1.0)])
    with pytest.raises(ValueError, match="all-mixture"):
        fit_ols(design, np.zeros(design.n_runs), model="scheffe-linear")


def test_mixture_anova_table_groups_linear_blending(components: list[MixtureFactor]):
    design, y = _yarn_design_and_response(components)
    result = fit_ols(design, y, model="scheffe-quadratic")
    table = anova_table(result, design, y)

    assert table.index[0] == "Linear blending"
    assert table.loc["Linear blending", "df"] == 2  # k - 1
    cross_terms = [name for name in result.term_names if ":" in name]
    assert list(table.index) == ["Linear blending", *cross_terms, "Residual", "Total"]

    # the decomposition adds up against the mean-corrected total
    ss_terms = table["SS"].iloc[:-2].sum() + table.loc["Residual", "SS"]
    assert ss_terms == pytest.approx(table.loc["Total", "SS"])
    df_terms = table["df"].iloc[:-2].sum() + table.loc["Residual", "df"]
    assert df_terms == pytest.approx(table.loc["Total", "df"])
    assert table.loc["Total", "df"] == design.n_runs - 1


# --------------------------------------------------------------------------- #
# D-optimal mixture designs (Phase 3 engine over mixture candidates)
# --------------------------------------------------------------------------- #


def test_d_optimal_mixture_design_sums_to_one_and_beats_centroid(
    components: list[MixtureFactor],
):
    # even resolution so the lattice contains the binary 0.5/0.5 blends the
    # quadratic blending model's optimal support needs
    region = mixture_candidates(components, resolution=4)
    design = d_optimal(
        components, n_runs=7, model="quadratic", region=region, n_restarts=5, seed=0
    )
    sums = design.runs[[f.name for f in components]].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)

    centroid = simplex_centroid(components)  # also 7 runs
    score_optimal = log_det_information(build_model_matrix(design, order=2).X)
    score_centroid = log_det_information(build_model_matrix(centroid, order=2).X)
    assert score_optimal >= score_centroid - 1e-9


def test_d_optimal_mixture_defaults_to_mixture_candidates(
    components: list[MixtureFactor],
):
    design = d_optimal(components, n_runs=6, model="quadratic", n_restarts=2, seed=1)
    sums = design.runs.to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)


def test_candidate_grid_rejects_mixture_factors(components: list[MixtureFactor]):
    with pytest.raises(TypeError, match="mixture_candidates"):
        doe.candidate_grid(components)
