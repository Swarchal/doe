"""Classical blocking generators (Phase 5c).

Blocking removes a known nuisance source (day, batch, operator) by grouping runs so the nuisance
is constant within a block and treatment comparisons are made within blocks. Here the block is
carried as a **reserved** ``CategoricalFactor`` named ``"block"`` (levels ``"B1"``, ``"B2"``, ...)
inside the :class:`~doe.factors.FactorSet`, so the existing deviation (effect) coding in
:func:`~doe.analysis.model.build_model_matrix` fits it with zero analysis changes -- the block
enters the model like any categorical factor.

Generators:
    * :func:`randomized_complete_block` -- every treatment once per block
    * :func:`latin_square`              -- each treatment once per row and once per column
    * :func:`blocked_factorial`         -- a ``2^k`` factorial confounded into ``2^q`` blocks

Within-block run order is randomized via the shared ``Design.randomize(within="block")`` machinery
(blocks stay contiguous and in order; runs shuffle inside each).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..design import Design
from ..factors import CategoricalFactor, ContinuousFactor, Factor, FactorSet
from .factorial import _generator_spec, _letter_to_name, full_factorial

BLOCK = "block"


def _block_levels(n: int) -> tuple[str, ...]:
    """Block level labels ``("B1", "B2", ...)``."""
    return tuple(f"B{i + 1}" for i in range(n))


def _require_no_block_collision(names: Sequence[str]) -> None:
    if BLOCK in names:
        raise ValueError(f"a factor named {BLOCK!r} collides with the reserved block column")


def randomized_complete_block(
    treatments: Sequence[Factor] | int, *, n_blocks: int, seed: int | None = None
) -> Design:
    """Randomized complete block design: every treatment appears once in every block.

    ``treatments`` is either a sequence of factors -- whose full-factorial crossing is the
    treatment set -- or an integer ``t``, which creates a single ``CategoricalFactor("treatment",
    ("T1", ..., "Tt"))``. Each block contains every treatment exactly once; the run order *within*
    each block is randomized independently (blocks stay contiguous and in order). The block is a
    reserved ``CategoricalFactor("block", ("B1", ...))``.

    Raises:
        ValueError: for ``n_blocks < 2``, an integer ``treatments < 2``, or a factor already
            named ``"block"``.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    if isinstance(treatments, int):
        if treatments < 2:
            raise ValueError("need at least 2 treatments")
        treatment_factor = CategoricalFactor(
            "treatment", tuple(f"T{i + 1}" for i in range(treatments))
        )
        treatment_runs = pd.DataFrame({"treatment": list(treatment_factor.levels)})
        treatment_factors: list[Factor] = [treatment_factor]
    else:
        treatment_factors = list(treatments)
        _require_no_block_collision([f.name for f in treatment_factors])
        treatment_runs = full_factorial(treatment_factors).runs
    _require_no_block_collision([f.name for f in treatment_factors])

    block_factor = CategoricalFactor(BLOCK, _block_levels(n_blocks))
    fs = FactorSet([*treatment_factors, block_factor])

    frames = []
    for b in range(n_blocks):
        block = treatment_runs.copy()
        block[BLOCK] = _block_levels(n_blocks)[b]
        frames.append(block)
    runs = pd.concat(frames, ignore_index=True)[fs.names]

    design = Design(
        runs,
        fs,
        name=f"rcb_{len(treatment_runs)}x{n_blocks}",
        meta={
            "generator": _generator_spec(
                "randomized_complete_block",
                treatments=treatments if isinstance(treatments, int) else "factors",
                n_blocks=n_blocks,
                seed=seed,
            )
        },
    )
    return design.randomize(seed, within=BLOCK)


def latin_square(treatments: int, *, seed: int | None = None) -> Design:
    """Latin square: a ``k×k`` design with each treatment once per row and once per column.

    Rows and columns are two crossed blocking directions (two nuisance factors). Built from the
    cyclic standard square ``L[i, j] = (i + j) mod k`` with seeded random row, column, and symbol
    permutations, yielding ``k²`` runs over three categorical factors ``row`` / ``column`` /
    ``treatment``.

    Raises:
        ValueError: for ``treatments < 2``.
    """
    k = treatments
    if k < 2:
        raise ValueError("a Latin square needs at least 2 treatments")
    rng = np.random.default_rng(seed)
    row_perm = rng.permutation(k)
    col_perm = rng.permutation(k)
    sym_perm = rng.permutation(k)

    row_factor = CategoricalFactor("row", tuple(f"R{i + 1}" for i in range(k)))
    col_factor = CategoricalFactor("column", tuple(f"C{j + 1}" for j in range(k)))
    treat_factor = CategoricalFactor("treatment", tuple(f"T{t + 1}" for t in range(k)))
    fs = FactorSet([row_factor, col_factor, treat_factor])

    rows_out: list[dict[str, object]] = []
    for a in range(k):
        for b in range(k):
            symbol = int(sym_perm[(row_perm[a] + col_perm[b]) % k])
            rows_out.append(
                {
                    "row": f"R{a + 1}",
                    "column": f"C{b + 1}",
                    "treatment": f"T{symbol + 1}",
                }
            )
    runs = pd.DataFrame(rows_out, columns=fs.names)
    return Design(
        runs,
        fs,
        name=f"latin_square_{k}",
        meta={"generator": _generator_spec("latin_square", treatments=k, seed=seed)},
    )


def _generalized_interactions(generator_letter_sets: list[frozenset[str]]) -> list[str]:
    """All non-empty products of the block generators (the full confounded set for ``q > 1``).

    A product of two contrasts is the symmetric difference of their factor-letter sets (repeated
    letters cancel, ``x²=1``). Returned as sorted letter strings, in generator-subset order.
    """
    confounded: list[str] = []
    q = len(generator_letter_sets)
    for r in range(1, q + 1):
        for combo in itertools.combinations(range(q), r):
            product: frozenset[str] = frozenset()
            for i in combo:
                product = product ^ generator_letter_sets[i]
            if product:
                confounded.append("".join(sorted(product)))
    # de-duplicate while preserving first-appearance order
    return list(dict.fromkeys(confounded))


def blocked_factorial(
    factors: Sequence[Factor], *, block_generators: Sequence[str], seed: int | None = None
) -> Design:
    """A ``2^k`` full factorial split into ``2^q`` blocks by confounding defining contrasts.

    Each string in ``block_generators`` (e.g. ``"ABC"``) names an interaction contrast to confound
    with blocks; the ``q`` contrasts' joint sign pattern assigns each run to one of ``2^q`` blocks
    (a reserved ``CategoricalFactor("block", ...)``). The confounded effects -- the generators
    **and all their generalized interactions** -- become inestimable and are recorded in
    ``meta["confounded_with_blocks"]`` (surfaced, not hidden). Within-block order is randomized.

    Raises:
        ValueError: for non-two-level factors, an empty ``block_generators``, a generator naming
            an unknown factor, or a factor already named ``"block"``.
    """
    fs = FactorSet(factors)
    _require_no_block_collision(fs.names)
    bad = [
        f.name
        for f in fs
        if not (
            isinstance(f, ContinuousFactor)
            or (isinstance(f, CategoricalFactor) and len(f.levels) == 2)
        )
    ]
    if bad:
        raise ValueError(f"blocked_factorial needs two-level factors; offending: {bad}")
    if not block_generators:
        raise ValueError("blocked_factorial needs at least one block generator")

    base = full_factorial(factors)
    coded = base.coded().to_numpy(dtype=float)  # (+/-1) columns in factor order

    letter_sets: list[frozenset[str]] = []
    contrasts: list[np.ndarray] = []
    for gen in block_generators:
        col = np.ones(coded.shape[0])
        letters: set[str] = set()
        for letter in gen:
            try:
                name = _letter_to_name(letter, fs.names)
            except ValueError as exc:
                raise ValueError(
                    f"block generator {gen!r} names unknown factor {letter!r}"
                ) from exc
            col = col * coded[:, fs.names.index(name)]
            letters.add(letter.upper())
        letter_sets.append(frozenset(letters))
        contrasts.append(col)

    # each run's sign vector over the q contrasts -> a block index in 0 .. 2^q - 1
    sign_bits = np.column_stack([(c > 0).astype(int) for c in contrasts])
    powers = 2 ** np.arange(sign_bits.shape[1])
    block_index = sign_bits @ powers
    unique_blocks = list(dict.fromkeys(block_index.tolist()))
    remap = {b: i for i, b in enumerate(unique_blocks)}
    block_labels = _block_levels(len(unique_blocks))
    block_column = [block_labels[remap[int(b)]] for b in block_index]

    block_factor = CategoricalFactor(BLOCK, block_labels)
    out_fs = FactorSet([*fs, block_factor])
    runs = base.runs.copy()
    runs[BLOCK] = block_column
    runs = runs[out_fs.names]

    confounded = _generalized_interactions(letter_sets)
    design = Design(
        runs,
        out_fs,
        name=f"blocked_factorial_{len(fs)}_{len(unique_blocks)}blocks",
        meta={
            "generator": _generator_spec(
                "blocked_factorial", block_generators=list(block_generators), seed=seed
            ),
            "confounded_with_blocks": confounded,
        },
    )
    return design.randomize(seed, within=BLOCK)
