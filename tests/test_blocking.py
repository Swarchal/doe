"""Classical blocking generators (Phase 5c).

Anchors: a randomized complete block has every treatment exactly once per block; a ``k×k`` Latin
square has each treatment once per row and once per column; a blocked factorial confounds the
requested contrast with blocks (the block column *is* the contrast numerically), records the full
confounded set (generators + generalized interactions), and the confounded effect is inestimable
in a fit while the rest are clean.
"""

import numpy as np
import pytest

from doe.analysis.model import build_model_matrix, coded_design_points
from doe.factors import ContinuousFactor
from doe.generators.blocking import (
    blocked_factorial,
    latin_square,
    randomized_complete_block,
)


def test_rcb_every_treatment_once_per_block():
    d = randomized_complete_block(4, n_blocks=3, seed=0)
    assert d.n_runs == 12
    assert sorted(set(d.runs["block"])) == ["B1", "B2", "B3"]
    for block in ["B1", "B2", "B3"]:
        treatments = sorted(d.runs.loc[d.runs["block"] == block, "treatment"])
        assert treatments == ["T1", "T2", "T3", "T4"]
    assert d.meta["randomized"] is True  # within-block order was randomized


def test_rcb_from_factor_crossing():
    factors = [ContinuousFactor("temp", 40, 80), ContinuousFactor("time", 5, 15)]
    d = randomized_complete_block(factors, n_blocks=2)
    # full-factorial crossing = 4 treatments, once per block => 8 runs
    assert d.n_runs == 8
    assert "block" in d.factors.names
    for block in ["B1", "B2"]:
        assert (d.runs["block"] == block).sum() == 4


def test_rcb_validates():
    with pytest.raises(ValueError, match="n_blocks"):
        randomized_complete_block(3, n_blocks=1)
    with pytest.raises(ValueError, match="block"):
        randomized_complete_block([ContinuousFactor("block", 0, 1)], n_blocks=2)


def test_latin_square_each_treatment_once_per_row_and_column():
    ls = latin_square(5, seed=1)
    assert ls.n_runs == 25
    pivot = ls.runs.pivot(index="row", columns="column", values="treatment")
    for r in pivot.index:
        assert len(set(pivot.loc[r])) == 5
    for c in pivot.columns:
        assert len(set(pivot[c])) == 5


def test_latin_square_validates():
    with pytest.raises(ValueError, match="at least 2"):
        latin_square(1)


def test_blocked_factorial_confounds_requested_contrast():
    factors = [ContinuousFactor(c, -1, 1) for c in "abc"]
    d = blocked_factorial(factors, block_generators=["ABC"], seed=2)
    assert d.n_runs == 8
    assert d.meta["confounded_with_blocks"] == ["ABC"]
    assert len(set(d.runs["block"])) == 2

    mm = build_model_matrix(d, order=1, interactions=True)
    pts = coded_design_points(d)  # a, b, c, block (numeric)
    abc = pts[:, 0] * pts[:, 1] * pts[:, 2]
    block_col = mm.X[:, mm.term_names.index("block[B2]")]
    # the block contrast column IS the ABC contrast (up to sign) -> they are confounded
    assert np.allclose(block_col, abc) or np.allclose(block_col, -abc)


def test_blocked_factorial_records_generalized_interactions():
    factors = [ContinuousFactor(c, -1, 1) for c in "abcd"]
    d = blocked_factorial(factors, block_generators=["ABC", "BCD"])
    # ABC * BCD = AD (the shared B, C cancel); the full confounded set is surfaced
    assert set(d.meta["confounded_with_blocks"]) == {"ABC", "BCD", "AD"}
    assert len(set(d.runs["block"])) == 4


def test_blocked_factorial_confounded_effect_is_inestimable_others_clean():
    # The confounded ABC contrast equals the block column, so it cannot be estimated separately:
    # appending it to the [main effects + 2FI + block] matrix does not raise the rank. Meanwhile
    # the block-inclusive model is full rank and recovers injected main effects cleanly.
    from doe.analysis.fit import fit_ols

    factors = [ContinuousFactor(c, -1, 1) for c in "abc"]
    d = blocked_factorial(factors, block_generators=["ABC"])
    mm = build_model_matrix(d, order=1, interactions=True)
    pts = coded_design_points(d)
    abc = (pts[:, 0] * pts[:, 1] * pts[:, 2]).reshape(-1, 1)
    base_rank = np.linalg.matrix_rank(mm.X)
    with_abc_rank = np.linalg.matrix_rank(np.hstack([mm.X, abc]))
    assert with_abc_rank == base_rank  # ABC adds no new information -> inestimable

    coded = pts[:, :3]
    y = 10 + 2 * coded[:, 0] - 1.5 * coded[:, 1] + 0.5 * coded[:, 2]
    fit = fit_ols(d.with_response("y", y), "y", interactions=False)
    summary = fit.summary()
    assert summary.loc["a", "coefficient"] == pytest.approx(2.0, abs=1e-6)
    assert summary.loc["b", "coefficient"] == pytest.approx(-1.5, abs=1e-6)
    assert summary.loc["c", "coefficient"] == pytest.approx(0.5, abs=1e-6)


def test_blocked_factorial_validates():
    with pytest.raises(ValueError, match="block generator"):
        blocked_factorial([ContinuousFactor(c, -1, 1) for c in "ab"], block_generators=["XYZ"])
    with pytest.raises(ValueError, match="at least one"):
        blocked_factorial([ContinuousFactor(c, -1, 1) for c in "ab"], block_generators=[])
