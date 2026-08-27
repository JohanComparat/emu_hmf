r"""What is being fitted: the CSST mass function, in *this* variance convention.

Two halves, and keeping them apart is the point.

**The target** is ``CEmulator``'s emulated :math:`\dd n/\dd\ln M` --- a Gaussian
process trained on the CSST suite, so it carries the simulations' calibration
rather than a fit to them.  It is numpy and not differentiable, which is fine:
it is the training target, evaluated offline, exactly as CLASS is for
:mod:`emu_pk`.

**The variance** is not the emulator's.  :math:`\sigma(M)` comes from
``ggah_mod`` on ``emu_pk``'s spectrum --- the *cold* field against
:math:`\bar\rho_{cb}` --- because that is the :math:`\sigma(M)` the recalibrated
fit will be evaluated with.  Fitting :math:`f(\sigma)` against one variance and
using it with another is the mismatch that makes a multiplicity function look
wrong when the convention around it is what moved, and this package exists
partly because that mismatch is easy to make.

Which mass definition
---------------------

``CEmulator`` offers ``RockstarM200m``, ``FoFM200c`` and ``RockstarMvir``, and
its wrapper in ``ggah_mod`` already records that these are genuinely different
masses rather than conventions.  The feedback that prompted this package asks
for :math:`200{\rm c}` --- but the only :math:`200{\rm c}` on offer is a
*friends-of-friends* mass, and pairing a FoF mass with a spherical-overdensity
multiplicity function is the category error ``ggah_mod.halos.calibration``
refuses one rung down.

So the default here is ``RockstarM200m``: a true SO mass, at the definition
``tinker08`` was itself calibrated in, and :math:`200{\rm c}` is reached
afterwards through the published :math:`\log\Delta` interpolation exactly as
``ggah_mod`` already does it.  ``FoFM200c`` is selectable and documented as
carrying a finder change along with the definition, because a package that
silently picked one of the two would be making that decision for its caller.
"""

from __future__ import annotations

import numpy as np

from . import box

__all__ = ["MASSDEFS", "DEFAULT_MASSDEF", "Z_TRAINED", "M_TRUSTED",
           "csst_dndlnM",
           "csst_tinker08", "set_cosmology", "sigma_chain",
           "to_ggah_cosmology"]

#: The three the emulator was trained on, and what each one *is*.
MASSDEFS = {
    "RockstarM200m": "spherical overdensity, 200 x mean, Rockstar",
    "FoFM200c": "friends-of-friends tuned to 200 x critical -- a different "
                "halo finder, not only a different boundary",
    "RockstarMvir": "spherical overdensity, virial, Rockstar",
}

#: The one that pairs with ``tinker08`` without changing halo finder as well.
DEFAULT_MASSDEF = "RockstarM200m"

#: Redshifts the emulator was trained at; it interpolates between them, so ``z``
#: is an input the recalibration gets nearly free -- as it was for ``emu_pk``.
Z_TRAINED = (0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

#: Where the emulator is smooth enough to be fitted, in :math:`M_\odot/h`.
#:
#: **Measured, not chosen.**  The quantity to be learned is the ratio of the
#: emulated mass function to the emulator's own Tinker08, and the residual of a
#: cubic in :math:`\ln M` through that ratio at the Planck fiducial is
#:
#: ==================  ==========
#: upper limit         residual
#: ==================  ==========
#: :math:`10^{14}`     4.2e-3
#: :math:`10^{14.5}`   1.4e-2
#: :math:`10^{15}`     3.0e-2
#: :math:`10^{15.5}`   4.0e-2
#: ==================  ==========
#:
#: so the target is smooth to a few parts in a thousand up to
#: :math:`10^{14}\,M_\odot/h` and progressively rougher above it, dropping to a
#: ratio of 0.91 by :math:`10^{15.5}`.  That is where a simulation suite runs
#: out of clusters, and a Gaussian process is noisiest where its training data
#: is thinnest -- so it is a property of the *target*, not of any fit made to
#: it.  A recalibration claimed to a per cent above this range would be
#: claiming to reproduce the emulator's own noise.
M_TRUSTED = (1e12, 1e14)


def to_ggah_cosmology(theta):
    """CSST's eight -> a ``ggah_mod`` ``Cosmology``.

    The conversions are all conventional and all easy to get wrong once:
    ``H0`` is in km/s/Mpc where ``ggah_mod`` wants ``h``; ``A`` is
    :math:`10^{9}A_s` where ``ggah_mod`` wants :math:`\\ln(10^{10}A_s)`; and
    CSST's ``Omegam`` includes the massive neutrinos, as ``ggah_mod``'s does.
    """
    from ggah_mod.cosmology import Cosmology
    d = dict(zip(box.PARAMS, np.asarray(theta, dtype=float)))
    return Cosmology.create(
        Omega_m=d["Omegam"], Omega_b=d["Omegab"], h=d["H0"] / 100.0,
        n_s=d["ns"], ln10A_s=float(np.log(10.0 * d["A"])),
        sum_mnu=d["mnu"], w0=d["w"], wa=d["wa"])


def set_cosmology(emu, theta):
    r"""Hand ``CEmulator`` a cosmology, in the dialect ``set_cosmos`` speaks.

    Not the one ``param_limits`` speaks, which is the trap: the bounds are
    stated in :math:`\Omega_m` and :math:`10^{9}A_s`, and the setter wants the
    *cold* density and :math:`A_s` itself, which it then multiplies by
    :math:`10^{9}` to check against those same bounds.  And the cold density is
    :math:`\Omega_{cdm}`, with the neutrinos excluded --- not
    :math:`\Omega_m - \Omega_b`, which still carries them.  Both conversions are
    one line, both are silently wrong if guessed, and ``ggah_mod``'s
    ``Cosmology`` already derives the second, so it derives it here too rather
    than being re-derived.
    """
    d = dict(zip(box.PARAMS, np.asarray(theta, dtype=float)))
    c = to_ggah_cosmology(theta)
    emu.set_cosmos(Omegab=float(c.Omega_b), Omegac=float(c.Omega_cdm),
                   H0=float(c.h) * 100.0, As=d["A"] * 1e-9, ns=float(c.n_s),
                   w=float(c.w0), wa=float(c.wa), mnu=float(c.sum_mnu))
    return emu


def csst_dndlnM(theta, z, m, massdef: str = DEFAULT_MASSDEF, emu=None):
    r""":math:`\dd n/\dd\ln M` from the CSST emulator [(Mpc/h)^-3].

    ``m`` in :math:`M_\odot/h`.  Refuses a cosmology outside the box rather
    than letting the Gaussian process extrapolate, which it will do silently.
    """
    if massdef not in MASSDEFS:
        raise ValueError(f"massdef must be one of {sorted(MASSDEFS)}, "
                         f"got {massdef!r}")
    box.check(dict(zip(box.PARAMS, np.asarray(theta, dtype=float))))
    emu = _emulator(theta) if emu is None else set_cosmology(emu, theta)
    return np.asarray(emu.get_dndlnM(z=np.atleast_1d(z), M=np.asarray(m),
                                     massdef=massdef))


def csst_tinker08(theta, z, m, delta_mean=200.0, emu=None):
    r"""``CEmulator``'s *own* Tinker08, at an explicit :math:`\Delta_{\rm m}`.

    Against the cold spectrum, which is this package's convention too.  It is
    here because the emulator shipping both its simulation-calibrated answer and
    its own analytic one makes the thing to be learned a *ratio* --- and a ratio
    is testable at the calibration cosmology in a way a fit from scratch is not.
    """
    box.check(dict(zip(box.PARAMS, np.asarray(theta, dtype=float))))
    emu = _emulator(theta) if emu is None else set_cosmology(emu, theta)
    return np.asarray(emu.get_dndlnM_Tinker08(
        z=np.atleast_1d(z), M=np.asarray(m), Pcb=True,
        Delta=float(delta_mean), rho_type="matter"))


def _emulator(theta=None):
    """``HMF_CEmulator``, with a cosmology set and ``ggah_mod``'s shim applied.

    The shim is not optional and not a version check.  Two of CEmulator's
    methods assign a size-1 array into a scalar slot, which numpy 2 refuses, so
    the mass function raises before returning anything.  ``ggah_mod`` carries
    the patch already --- *probed* by calling the method and catching the error,
    so a fixed upstream stops being patched --- and it has to be applied after a
    cosmology is set, because that is what makes the probe reach the failure.
    """
    from ggah_mod.halos._cemulator_compat import ensure_cemulator_works
    from CEmulator.Emulator import HMF_CEmulator
    emu = HMF_CEmulator()
    set_cosmology(emu, _FIDUCIAL if theta is None else theta)
    ensure_cemulator_works(emu)
    if theta is not None:
        set_cosmology(emu, theta)
    return emu


#: A point in the middle of the box, used only to probe the shim.
_FIDUCIAL = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.06])


def sigma_chain(theta, z, m, pk=None):
    r"""``(sigma, dlnsigma_dlnM, rho_cb)`` in ``ggah_mod``'s convention.

    The cold field against :math:`\bar\rho_{cb}`, and the logarithmic derivative
    by automatic differentiation of the same integral rather than by
    differencing it -- both because that is what the recalibrated fit will be
    evaluated with, and because a finite difference of a quadrature is how
    percent-level noise gets into a mass function.
    """
    import jax.numpy as jnp
    from ggah_mod.halos.variance import sigma_of_mass, dln_sigma_dln_mass

    cosmo = to_ggah_cosmology(theta)
    if pk is None:
        from ggah_mod.cosmology.power import make_pk
        pk = make_pk("emu_pk")
    k = np.logspace(-4, np.log10(200.0), 512)
    p_cb = pk.pk_cb(k, float(z), cosmo)
    m = jnp.asarray(m)
    sig = sigma_of_mass(m, k, p_cb, cosmo.rho_cold)
    dlns = dln_sigma_dln_mass(m, k, p_cb, cosmo.rho_cold)
    return np.asarray(sig), np.asarray(dlns), float(cosmo.rho_cold)
