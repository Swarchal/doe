"""Phase 3b: optimal designs via coordinate exchange (D-/I-optimal, augment)."""

import numpy as np
import pytest

from doe.analysis.diagnostics import efficiency, log_det_information
from doe.analysis.model import build_model_matrix, expand_coded_points
from doe.design import Design
from doe.factors import CategoricalFactor, ContinuousFactor
from doe.generators.factorial import full_factorial
from doe.generators.optimal import (
    OptimalDesign,
    _model_spec,
    augment,
    candidate_grid,
    coordinate_exchange,
    d_optimal,
    i_optimal,
)


def _factors(n):
    return [ContinuousFactor(chr(ord("a") + i), low=-1.0, high=1.0) for i in range(n)]


def _log_det(design, *, order, interactions):
    x = build_model_matrix(design, order=order, interactions=interactions).X
    return log_det_information(x)


# --------------------------------------------------------------------------- #
# Candidate region
# --------------------------------------------------------------------------- #


def test_candidate_grid_shape_and_levels():
    grid = candidate_grid(_factors(2), levels=3)
    assert grid.shape == (9, 2)  # 3 levels ^ 2 factors
    assert set(np.unique(grid)).issubset({-1.0, 0.0, 1.0})


def test_candidate_grid_uses_categorical_level_count():
    factors = [ContinuousFactor("x", 0.0, 1.0), CategoricalFactor("cat", ("A", "B", "C"))]
    grid = candidate_grid(factors, levels=2)
    assert grid.shape == (6, 2)  # 2 continuous levels * 3 categorical levels
    assert set(np.unique(grid[:, 0])) == {-1.0, 1.0}
    assert set(np.unique(grid[:, 1])) == {-1.0, 0.0, 1.0}


def test_candidate_grid_rejects_too_few_levels():
    with pytest.raises(ValueError, match="at least 2"):
        candidate_grid(_factors(1), levels=1)


# --------------------------------------------------------------------------- #
# Coordinate exchange engine
# --------------------------------------------------------------------------- #


def test_model_spec_normalizes_supported_model_names():
    assert _model_spec("linear") == (1, True)
    assert _model_spec("quadratic") == (2, True)


def test_coordinate_exchange_rejects_unknown_model_before_search():
    with pytest.raises(ValueError, match="unknown model"):
        coordinate_exchange(_factors(2), n_runs=4, model="cubic")  # type: ignore[arg-type]


def test_coordinate_exchange_returns_report():
    result = coordinate_exchange(
        _factors(2), n_runs=4, model="linear", criterion="D", seed=0
    )
    assert isinstance(result, OptimalDesign)
    assert isinstance(result.design, Design)
    assert result.criterion == "D"
    assert result.design.n_runs == 4


def test_coordinate_exchange_is_reproducible_under_seed():
    a = coordinate_exchange(_factors(2), n_runs=9, model="quadratic", seed=42)
    b = coordinate_exchange(_factors(2), n_runs=9, model="quadratic", seed=42)
    assert np.allclose(a.design.coded().to_numpy(), b.design.coded().to_numpy())
    assert a.score == pytest.approx(b.score)


def test_rank1_update_agrees_with_full_refit():
    # the reported determinant must match a from-scratch slogdet of the final design
    result = coordinate_exchange(_factors(2), n_runs=9, model="quadratic", seed=0)
    recomputed = _log_det(result.design, order=2, interactions=True)
    assert result.score == pytest.approx(recomputed)


def test_coordinate_exchange_supports_categorical_candidates():
    factors = [ContinuousFactor("temp", 0.0, 100.0), CategoricalFactor("cat", ("A", "B", "C"))]

    result = coordinate_exchange(factors, n_runs=6, model="linear", criterion="D", seed=0)

    assert set(result.design.runs["cat"]).issubset({"A", "B", "C"})
    recomputed = _log_det(result.design, order=1, interactions=True)
    assert result.score == pytest.approx(recomputed)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"criterion": "A"}, "criterion"),  # type: ignore[dict-item]
        ({"n_runs": 0}, "n_runs"),
        ({"n_restarts": 0}, "n_restarts"),
        ({"max_iter": 0}, "max_iter"),
    ],
)
def test_coordinate_exchange_rejects_invalid_scalar_options(kwargs, message):
    params = {"n_runs": 4, "model": "linear", **kwargs}
    with pytest.raises(ValueError, match=message):
        coordinate_exchange(_factors(2), **params)  # type: ignore[arg-type]


def test_coordinate_exchange_rejects_bad_regions():
    factors = _factors(2)
    with pytest.raises(ValueError, match="2-D"):
        coordinate_exchange(factors, n_runs=4, model="linear", region=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="factors has 2 entries"):
        coordinate_exchange(factors, n_runs=4, model="linear", region=np.zeros((3, 1)))
    with pytest.raises(ValueError, match="at least one"):
        coordinate_exchange(factors, n_runs=4, model="linear", region=np.empty((0, 2)))
    with pytest.raises(ValueError, match="finite"):
        coordinate_exchange(factors, n_runs=4, model="linear", region=np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        coordinate_exchange(factors, n_runs=4, model="linear", region=np.array([[0.0, 1.5]]))


def test_coordinate_exchange_rejects_bad_fixed_runs():
    factors = _factors(2)
    with pytest.raises(ValueError, match="2-D"):
        coordinate_exchange(factors, n_runs=4, model="linear", fixed_runs=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="factors has 2 entries"):
        coordinate_exchange(factors, n_runs=4, model="linear", fixed_runs=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="more rows"):
        coordinate_exchange(factors, n_runs=4, model="linear", fixed_runs=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="finite"):
        coordinate_exchange(factors, n_runs=4, model="linear", fixed_runs=np.array([[0.0, np.inf]]))


def test_coordinate_exchange_requires_enough_runs_for_model_terms():
    with pytest.raises(ValueError, match="cannot estimate"):
        coordinate_exchange(_factors(2), n_runs=5, model="quadratic")


# --------------------------------------------------------------------------- #
# D-optimal
# --------------------------------------------------------------------------- #


def test_d_optimal_recovers_full_factorial():
    # for the first-order + interaction model with n_runs == 2^k, the D-optimal design is the
    # full factorial (up to a relabeling of runs): same |XtX|, D-efficiency 1.
    k = 3
    design = d_optimal(_factors(k), n_runs=2**k, model="linear")
    got = _log_det(design, order=1, interactions=True)
    expected = _log_det(full_factorial(_factors(k)), order=1, interactions=True)
    assert got == pytest.approx(expected)
    assert design.meta["criterion"] == "D"
    assert design.meta["d_efficiency"] == pytest.approx(1.0)


def test_d_optimal_quadratic_k2_n9_matches_known_determinant():
    # k=2 quadratic, 9 runs: the optimum is the 3^2 grid (a face-centered design); pin |XtX|.
    design = d_optimal(_factors(2), n_runs=9, model="quadratic")
    reference = full_factorial(_factors(2), levels=3)
    got = _log_det(design, order=2, interactions=True)
    expected = _log_det(reference, order=2, interactions=True)
    assert got == pytest.approx(expected, rel=1e-6)


def test_d_optimal_reproducible_under_seed():
    # the seed forwarded through the wrapper to coordinate_exchange makes the run deterministic
    a = d_optimal(_factors(2), n_runs=9, model="quadratic", seed=0)
    b = d_optimal(_factors(2), n_runs=9, model="quadratic", seed=0)
    assert np.allclose(a.coded().to_numpy(), b.coded().to_numpy())


def test_unseeded_search_records_a_concrete_reusable_seed():
    # seed=None must not serialize as null: a concrete seed is drawn and recorded (as
    # Design.randomize does) so a serialized optimal design can always regenerate its search.
    a = d_optimal(_factors(2), n_runs=9, model="quadratic")
    recorded = a.meta["seed"]
    assert isinstance(recorded, int)
    b = d_optimal(_factors(2), n_runs=9, model="quadratic", seed=recorded)
    assert np.allclose(a.coded().to_numpy(), b.coded().to_numpy())


# --------------------------------------------------------------------------- #
# I-optimal
# --------------------------------------------------------------------------- #


def test_i_optimal_has_lower_average_prediction_variance_than_d_optimal():
    factors = _factors(2)
    region = candidate_grid(factors, levels=5)
    d = d_optimal(factors, n_runs=9, model="quadratic")
    i = i_optimal(factors, n_runs=9, model="quadratic")
    # I-efficiency is the (normalised) inverse of average prediction variance over the region:
    # higher is better. Evaluate BOTH designs on the *same* region -- comparing their stored
    # meta["score"] directly would be meaningless (D stores a log-det, I an average variance).
    d_eff = efficiency(d, order=2, interactions=True, region=region).i
    i_eff = efficiency(i, order=2, interactions=True, region=region).i
    assert i_eff >= d_eff - 1e-9
    assert i.meta["criterion"] == "I"
    assert i.meta["score"] > 0.0


def test_i_optimal_score_is_average_prediction_variance():
    factors = _factors(2)
    region = candidate_grid(factors, levels=5)
    result = coordinate_exchange(
        factors,
        n_runs=8,
        model="quadratic",
        criterion="I",
        region=region,
        seed=1,
        n_restarts=50,
    )

    design_points = result.design.coded().to_numpy(dtype=float)
    expanded = expand_coded_points(
        np.vstack([design_points, region]), result.design.factors, order=2, interactions=True
    )
    x, f_region = expanded.X[: result.design.n_runs], expanded.X[result.design.n_runs :]
    info_inv = np.linalg.inv(x.T @ x)
    expected = np.mean(np.einsum("ij,jk,ik->i", f_region, info_inv, f_region))

    assert result.score == pytest.approx(expected)
    assert result.design.meta["score"] == pytest.approx(expected)


def test_i_optimal_handles_singular_random_starts():
    # With a saturated 2-factor quadratic, duplicate random starts are singular. The search must
    # still exchange toward an estimable design and report a finite average prediction variance.
    factors = _factors(2)
    result = coordinate_exchange(
        factors,
        n_runs=6,
        model="quadratic",
        criterion="I",
        region=candidate_grid(factors, levels=3),
        seed=0,
        n_restarts=10,
    )

    assert np.isfinite(result.score)
    assert efficiency(result.design, order=2, interactions=True).d > 0.0


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


def test_augment_preserves_existing_rows_and_grows_information():
    base = full_factorial(_factors(2))
    n_existing = base.n_runs
    augmented = augment(base, n_runs=4, model="quadratic", criterion="D")

    assert augmented.n_runs == n_existing + 4
    # the original rows are kept byte-for-byte at the front
    base_coded = base.coded().to_numpy()
    aug_coded = augmented.coded().to_numpy()
    assert np.allclose(aug_coded[:n_existing], base_coded)
    # extra optimal runs never reduce information
    before = _log_det(base, order=2, interactions=True)
    after = _log_det(augmented, order=2, interactions=True)
    assert after >= before - 1e-9


def test_augment_tags_point_types():
    base = full_factorial(_factors(2))
    augmented = augment(base, n_runs=2, model="quadratic")
    assert augmented.point_types is not None
    assert set(augmented.point_types) == {"existing", "augment"}


def test_augment_supports_categorical_factors():
    factors = [
        ContinuousFactor("temp", 0.0, 100.0),
        CategoricalFactor("cat", ("A", "B", "C")),
    ]
    base = full_factorial(factors)

    augmented = augment(base, n_runs=2, model="linear", criterion="D", seed=0)

    assert augmented.n_runs == base.n_runs + 2
    assert augmented.runs["cat"].iloc[: base.n_runs].tolist() == base.runs["cat"].tolist()
    assert set(augmented.runs["cat"]).issubset({"A", "B", "C"})


def test_augment_rejects_invalid_arguments():
    base = full_factorial(_factors(2))
    with pytest.raises(ValueError, match="n_runs"):
        augment(base, n_runs=0)
    with pytest.raises(TypeError, match="fixed_runs"):
        augment(base, n_runs=2, fixed_runs=np.zeros((base.n_runs, 2)))


def test_augment_supports_i_optimal_criterion():
    base = full_factorial(_factors(2))
    augmented = augment(base, n_runs=4, model="quadratic", criterion="I", seed=1, n_restarts=20)

    assert augmented.meta["criterion"] == "I"
    assert augmented.meta["score"] > 0.0
    assert augmented.point_types == ("existing",) * base.n_runs + ("augment",) * 4
