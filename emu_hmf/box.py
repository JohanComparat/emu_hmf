r"""The hypercube the recalibration lives in, taken from CSST rather than chosen.

The target of this package is the CSSTemu mass function, so the box
is *its* box.  Copying the numbers here rather than importing them is
deliberate and is checked: CSSTemu belongs to the ``[gen]`` install, and a
package whose whole point is to be importable by a forecast cannot make the
forecast depend on a Gaussian-process emulator to find out what its own bounds
are.  ``tests/test_box.py`` asserts these against CSSTemu's own
``param_limits``, so the copy cannot drift without a test failing.

**Narrower than** :mod:`emu_pk`'s **in three axes**, which matters because the
:math:`\sigma(M)` this package recalibrates against comes from there:
:mod:`emu_pk` reaches :math:`h = 0.55` where CSST starts at :math:`H_0 = 60`,
and :math:`\Sigma m_\nu = 0.6` where CSST stops at :math:`0.3`.  Outside this
box the recalibration is undefined and says so, rather than extrapolating a fit
whose training data ends.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PARAMS", "BOX", "sample", "check", "inside"]

#: Column order.  CSSTemu's own names, so a design matrix built here can
#: be handed to it without a mapping that could be got wrong in one place.
PARAMS = ("Omegab", "Omegam", "H0", "ns", "A", "w", "wa", "mnu")

#: Closed bounds, inclusive, exactly as CSSTemu states them.
BOX = {
    "Omegab": (0.04, 0.06),
    "Omegam": (0.24, 0.40),
    "H0":     (60.0, 80.0),
    "ns":     (0.92, 1.00),
    "A":      (1.7, 2.5),
    "w":      (-1.3, -0.7),
    "wa":     (-0.5, 0.5),
    "mnu":    (0.0, 0.3),
}


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube on the unit cube, one stratified point per row."""
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(cut[:, j])
    return cut


def sample(n: int, seed: int = 20260828) -> np.ndarray:
    """``(n, 8)`` design in :data:`PARAMS` order.

    Deterministic in ``seed``: a design reproducible from a seed alone means a
    shard can be regenerated later without shipping the matrix, and two shards
    can never disagree about which index means which cosmology.

    Unlike :mod:`emu_pk` there is no ``w0 + wa < 0`` rejection here.  CSST's box
    already excludes the corner where dark energy dominates early --- its
    ``w`` stops at ``-0.7`` and its ``wa`` at ``0.5``, so ``w + wa <= -0.2``
    everywhere in it --- and adding a rejection that never fires would be a
    guard that looks like it is doing something.
    """
    rng = np.random.default_rng(seed)
    u = _lhs(n, len(PARAMS), rng)
    lo = np.array([BOX[p][0] for p in PARAMS])
    hi = np.array([BOX[p][1] for p in PARAMS])
    return lo + u * (hi - lo)


def inside(values: dict) -> dict:
    """``{name: (value, bounds)}`` for every named parameter out of bounds."""
    return {p: (float(v), BOX[p]) for p, v in values.items()
            if p in BOX and not BOX[p][0] <= float(v) <= BOX[p][1]}


def check(values: dict) -> None:
    """Raise naming every parameter outside the box, not just the first."""
    bad = inside(values)
    if bad:
        raise ValueError(
            "outside the CSST emulator's box, where this recalibration has no "
            "training data: "
            + "; ".join(f"{p} = {v:.5g} not in {b}" for p, (v, b) in bad.items())
            + ".  The fit is not defined there and will not be extrapolated.")
