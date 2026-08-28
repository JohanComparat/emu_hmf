r"""Inference: Tinker08 with cosmology-dependent parameters, in pure JAX.

Importable with numpy and jax alone.  Nothing here needs ``CEmulator``,
``classy``, ``ggah_mod`` or ``optax`` --- those built the thing, and a forecast
that only wants to *evaluate* it should not install them.

The form is unchanged
-----------------------

.. math::

    f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]
                e^{-c/\sigma^{2}}

with the published :math:`\Delta = 200` values and redshift evolution, and each
of the four multiplied by :math:`e^{g_i(\theta, z)}` where :math:`g` is a small
network.  Two properties follow from writing it that way rather than fitting
four free functions:

* :math:`g = 0` recovers Tinker08 **exactly**, so "the correction is zero" is a
  statement one can make and test rather than a limit one hopes for;
* the result is still a fit with named parameters, so a reader can ask what the
  recalibration did to the amplitude as against the tilt, which a black box
  cannot answer.
"""

from __future__ import annotations

import functools
import pathlib
import types

import jax
import jax.numpy as jnp
import numpy as np

from . import box

__all__ = ["T08", "tinker08", "HmfCorrection", "load_weights",
           "DEFAULT_WEIGHTS", "WEIGHTS"]

_DATA = pathlib.Path(__file__).resolve().parent / "data"

#: One weights file per halo definition, because the correction is *not* the
#: same function at two of them.
#:
#: That is measured rather than assumed.  Fitting the same architecture against
#: the emulator's ``RockstarMvir`` -- a different overdensity from the *same*
#: halo finder, so the comparison isolates the boundary -- gives a correction
#: that reduces the residual just as well (10.9 per cent to 0.54, against 7.0 to
#: 0.52 at 200m) and is a different function: over the peak heights both were
#: fitted on the two disagree by far more than the residual either achieves,
#: and by a margin comparable to the offset they both correct.
#:
#: So there is no single correction with a Delta argument, and pretending
#: otherwise would put an error the size of the correction into whichever
#: definition was not fitted.  Two files, two registry entries in ``ggah_mod``,
#: and a calibration guard that refuses to mix them.
WEIGHTS = {
    "200m": _DATA / "emu_hmf_mlp.npz",
    "vir": _DATA / "emu_hmf_mlp_vir.npz",
}
#: The 200m correction, which is what ``HmfCorrection()`` builds unasked.
DEFAULT_WEIGHTS = WEIGHTS["200m"]

#: Tinker et al. (2008) Table 2 at :math:`\Delta_{\rm m} = 200`, and the
#: published redshift evolution.  Restated here rather than imported from
#: ``ggah_mod`` because this module is the one that must import nothing ---
#: ``tests/test_model.py`` asserts this against the published table written out
#: independently, and against the halo-model code's own Tinker08 wherever that
#: is installed -- which is what makes the restatement safe.
T08 = {"A0": 0.186, "a0": 1.47, "b0": 2.57, "c0": 1.19,
       "Az": -0.14, "az": -0.06}

#: :math:`b`'s exponent is a function of :math:`\Delta`, not a constant:
#: :math:`\alpha = 10^{-(0.75/\log_{10}(\Delta/75))^{1.2}}`, which is
#: 0.0106 at :math:`\Delta = 200`.
T08_ALPHA_200 = float(10.0 ** (-((0.75 / np.log10(200.0 / 75.0)) ** 1.2)))


def tinker08(sigma, z=0.0, g=None):
    r"""Tinker08 at :math:`\Delta_{\rm m}=200`, optionally corrected.

    ``g`` is ``(4,)`` of log-corrections to :math:`(A, a, b, c)` in that order,
    or ``None`` for the published fit unchanged.
    """
    sigma = jnp.asarray(sigma)
    zp1 = 1.0 + jnp.asarray(z)
    A = T08["A0"] * zp1 ** T08["Az"]
    a = T08["a0"] * zp1 ** T08["az"]
    b = T08["b0"] * zp1 ** -T08_ALPHA_200
    c = jnp.asarray(T08["c0"])
    if g is not None:
        g = jnp.asarray(g)
        A, a, b, c = (A * jnp.exp(g[..., 0]), a * jnp.exp(g[..., 1]),
                      b * jnp.exp(g[..., 2]), c * jnp.exp(g[..., 3]))
    return A * ((sigma / b) ** -a + 1.0) * jnp.exp(-c / sigma ** 2)


def _mlp(params, x):
    """Tanh stack.  :math:`C^\\infty`, because this gets differentiated."""
    n = len(params) // 2
    for i in range(n - 1):
        x = jnp.tanh(x @ params[f"W{i}"] + params[f"b{i}"])
    return x @ params[f"W{n - 1}"] + params[f"b{n - 1}"]


def normalise(theta, z, z_max=3.0):
    """``(theta, z)`` -> the unit cube, in :data:`emu_hmf.box.PARAMS` order."""
    theta = jnp.asarray(theta)
    lo = jnp.array([box.BOX[p][0] for p in box.PARAMS])
    hi = jnp.array([box.BOX[p][1] for p in box.PARAMS])
    u = (theta - lo) / (hi - lo)
    zz = jnp.atleast_1d(jnp.asarray(z, dtype=float) / z_max)
    return jnp.concatenate([jnp.broadcast_to(u, zz.shape + u.shape[-1:]),
                            zz[..., None]], axis=-1)


@functools.lru_cache(maxsize=4)
def load_weights(path=None):
    """The arrays in a weights file, read once per path and then shared.

    Read-only, both ways round: the mapping is a proxy and the arrays have
    their write flag cleared.  The cache hands the *same* objects to every
    caller, so a mutation by one would silently change the correction every
    other caller evaluates -- and a network whose weights moved would still
    return entirely plausible numbers.
    """
    p = DEFAULT_WEIGHTS if path is None else pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing.  Fit it with `python -m emu_hmf.fit` on a "
            "generated training set, or point `HmfCorrection(weights=...)` "
            "at one.")
    with np.load(p) as d:
        w = {}
        for k in d.files:
            a = np.asarray(d[k])
            a.flags.writeable = False
            w[k] = a
    return types.MappingProxyType(w)


class HmfCorrection:
    r"""The recalibrated multiplicity function.

    ``check_box`` is on by default and skipped under tracing, exactly as
    :class:`emu_pk.model.PkEmulator` does it: the values are not available
    inside a ``jit``, and raising there would break the gradient this exists to
    provide.  A jitted forward model is checked once when it is built.
    """

    def __init__(self, weights=None, check_box: bool = True):
        w = load_weights(weights)
        self._p = {k: jnp.asarray(v) for k, v in w.items()
                   if k[0] in "Wb" and k[1:].isdigit()}
        self._check_box = bool(check_box)
        self.meta = {k: w[k] for k in w if k not in self._p}

    def _validate(self, theta):
        if not self._check_box:
            return
        vals = {}
        for p, v in zip(box.PARAMS, jnp.asarray(theta)):
            try:
                vals[p] = float(v)
            except Exception:                       # a tracer; nothing to check
                return
        box.check(vals)

    def g(self, theta, z):
        """The four log-corrections, shape ``(..., 4)``."""
        return _mlp(self._p, normalise(theta, z))

    def fsigma(self, sigma, theta, z=0.0):
        r""":math:`f(\sigma)` for this cosmology and redshift."""
        self._validate(theta)
        g = self.g(theta, z)
        g = g[0] if jnp.ndim(z) == 0 else g
        return tinker08(sigma, z, g)

    def dndlnM(self, m, sigma, dlnsigma_dlnm, rho_cold, theta, z=0.0):
        r""":math:`\dd n/\dd\ln M = f(\sigma)\,(\bar\rho_{cb}/M)\,
        |\dd\ln\sigma/\dd\ln M|`.

        The variance is passed in rather than computed: this package has no
        spectrum and should not acquire one, and the caller's :math:`\sigma(M)`
        is the one the fit was made against.
        """
        f = self.fsigma(sigma, theta, z)
        return f * (rho_cold / jnp.asarray(m)) * jnp.abs(jnp.asarray(dlnsigma_dlnm))
