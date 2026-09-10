r"""The hypercube the recalibration is defined on, taken from CSST rather than chosen.

The target of this package is the CSSTemu mass function, so the box is its box.
The numbers are copied here rather than imported, because a package meant to be
importable by a forecast cannot require a Gaussian-process emulator to state its
own bounds.  ``tests/test_box.py`` asserts the copy against CSSTemu's
``param_limits``.

**How it sits inside** :mod:`emu_pk`'s **box**, which is where the
:math:`\sigma(M)` this package is evaluated with comes from.  The two are stated
in different variables, so ``tests/test_box.py`` makes the comparison by mapping
this box into that one.

On seven of the eight axes they share, this box is the narrower one and sits
wholly inside: :mod:`emu_pk` reaches :math:`h = 0.55` where CSST starts at
:math:`H_0 = 60`, and :math:`\Sigma m_\nu = 0.6` where CSST stops at 0.3.

On the eighth it escapes.  CSST bounds :math:`\Omega_b` and :math:`H_0`
separately, so :math:`\omega_b = \Omega_b h^2` runs from 0.0147 to 0.0382 across
it against :mod:`emu_pk`'s 0.0170 to 0.0280, outside on both sides, and 30 per
cent of this box's design points sit there.  It is the only axis that excludes
any, and the reason the training set is generated with CLASS; see
:mod:`emu_hmf.generate`.

:mod:`emu_pk` 2.0.0 carries three axes this box does not have: :math:`\Omega_k`,
and the two ratios that divide :math:`\Sigma m_\nu` over three eigenstates.
CSST's suite is flat and its neutrinos are one species, so there is nothing here
to bound.  :data:`emu_hmf.target.OMEGA_K_COST` and
:data:`~emu_hmf.target.NU_ORDERING_COST` measure what that costs.

Outside this box the recalibration is undefined and refuses.
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
