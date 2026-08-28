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
           "NU_TRUSTED", "csst_dndlnM", "csst_tinker08", "set_cosmology",
           "sigma_chain", "to_ggah_cosmology", "theta_from_cosmology"]

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

#: Where the *fit* is made, in peak height rather than in mass.
#:
#: Also measured.  Against ``ggah_mod``'s own ``tinker08`` at the Planck
#: fiducial, the emulator's multiplicity function sits at
#:
#: ==================  =========================  =======
#: :math:`\nu` range   ratio                      median
#: ==================  =========================  =======
#: 0.5--2.0            0.983--1.055               1.018
#: 0.5--2.5            0.983--1.128               1.023
#: 0.5--3.0            0.983--1.229               1.027
#: 0.5--4.0            0.983--1.353               1.032
#: 0.5--6.0            0.983--2.192               1.036
#: ==================  =========================  =======
#:
#: so a few per cent up to :math:`\nu \simeq 3` and then the exponential tail,
#: where a per-cent error in :math:`\sigma` is a tens-of-per-cent error in
#: :math:`f`.  Peak height is the right variable for the cut because it unifies
#: mass and redshift: the same :math:`\nu = 3` is :math:`10^{15}\,M_\odot/h` at
#: :math:`z = 0` and :math:`3\times10^{13}` at :math:`z = 2`, and a cut in mass
#: alone would keep the tail at high redshift and discard signal at low.
#:
#: ``tinker08``'s own calibration stops at :math:`z = 2.5`
#: (``ggah_mod.halos.mass_function.CALIBRATION``), so the upper end of this
#: range is also roughly where the fit being corrected stops meaning anything.
NU_TRUSTED = (0.5, 3.0)


def to_ggah_cosmology(theta):
    r"""CSST's eight -> a ``ggah_mod`` ``Cosmology``.

    Two of the conversions are conventional and stated in the box's own
    documentation: ``H0`` is in km/s/Mpc where ``ggah_mod`` wants ``h``, and
    ``A`` is :math:`10^{9}A_s` where ``ggah_mod`` wants :math:`\ln(10^{10}A_s)`.
    The third is not stated anywhere and is the one that costs something.

    CSST's ``Omegam`` --- the symbol ``param_limits`` bounds --- is the **cold**
    density, :math:`\Omega_b + \Omega_{cdm}`, with the massive neutrinos
    excluded; ``CEmulator`` carries the total separately as ``Cosmo.OmegaM``.
    ``ggah_mod``'s :attr:`Omega_m` is the total.  So the two agree on
    :attr:`Omega_cb`, not on :attr:`Omega_m`, and the neutrino density has to be
    added on the way in.

    It is added by *asking ``ggah_mod``* rather than by dividing
    :math:`\Sigma m_\nu` by 93.14 eV here: :attr:`Omega_nu` is a
    :math:`\Sigma m_\nu`-and-:math:`h` quantity that does not depend on
    :attr:`Omega_m`, so one throwaway construction reads it off in whatever
    convention the package actually uses, and the second construction is exact
    by that package's own definition instead of by a constant repeated in two
    repositories.  Getting it wrong is worth half a per cent in :math:`\Omega_m`
    at 0.06 eV and two per cent at 0.3 --- small enough to survive every smoke
    test, and directly in the variance the correction is being fitted against.
    """
    from ggah_mod.cosmology import Cosmology
    d = dict(zip(box.PARAMS, np.asarray(theta, dtype=float)))
    kw = dict(Omega_b=d["Omegab"], h=d["H0"] / 100.0, n_s=d["ns"],
              ln10A_s=float(np.log(10.0 * d["A"])),
              sum_mnu=d["mnu"], w0=d["w"], wa=d["wa"])
    probe = Cosmology.create(Omega_m=d["Omegam"], **kw)
    return Cosmology.create(Omega_m=d["Omegam"] + float(probe.Omega_nu), **kw)


def theta_from_cosmology(cosmo):
    r"""A ``ggah_mod`` ``Cosmology`` -> CSST's eight, in :data:`box.PARAMS` order.

    The inverse of :func:`to_ggah_cosmology`, and it lives here for the same
    reason that one does: this package owns the convention, and a second
    implementation of it in ``ggah_mod`` would be a second thing to keep in
    step.  ``tests/test_target.py`` pins the round trip in both directions.

    Traceable, unlike its inverse.  ``to_ggah_cosmology`` builds a
    ``Cosmology`` from concrete numbers and can afford ``float()``; this one is
    called from inside the halo layer's differentiable path, where every value
    may be a tracer, so it does arithmetic in ``jnp`` and never asks for a
    concrete value.  The two directions genuinely need different treatment,
    which is why this is not a one-line inversion.

    The density is the trap, as it is going the other way: CSST's ``Omegam`` is
    the cold density, so it comes from :attr:`Omega_cb` and not from
    :attr:`Omega_m`.
    """
    import jax.numpy as jnp

    return jnp.stack([
        jnp.asarray(cosmo.Omega_b),
        jnp.asarray(cosmo.Omega_cb),            # cold, not total
        jnp.asarray(cosmo.h) * 100.0,
        jnp.asarray(cosmo.n_s),
        jnp.exp(jnp.asarray(cosmo.ln10A_s)) / 10.0,   # ln(1e10 A_s) -> 1e9 A_s
        jnp.asarray(cosmo.w0),
        jnp.asarray(cosmo.wa),
        jnp.asarray(cosmo.sum_mnu),
    ])


def set_cosmology(emu, theta):
    r"""Hand ``CEmulator`` a cosmology, in the dialect ``set_cosmos`` speaks.

    Not the one ``param_limits`` speaks, which is the trap: the bounds are
    stated in :math:`\Omega_m` and :math:`10^{9}A_s`, and the setter wants the
    *cold* density and :math:`A_s` itself, which it then multiplies by
    :math:`10^{9}` to check against those same bounds.  And the cold density it
    wants is :math:`\Omega_{cdm}` with the neutrinos excluded --- not
    :math:`\Omega_m - \Omega_b`, which still carries them; ``set_cosmos`` adds
    :math:`\Omega_\nu` back to reach the total, so handing it the wrong one
    double-counts.  ``ggah_mod``'s ``Cosmology`` already derives exactly this
    quantity as :attr:`Omega_cdm`, so it is named here rather than re-derived,
    and the two packages then agree symbol for symbol: CSST's bounded
    ``Omegam`` is ``ggah_mod``'s :attr:`Omega_cb`, and ``set_cosmos``'s
    ``Omegac`` is its :attr:`Omega_cdm`.
    """
    d = dict(zip(box.PARAMS, np.asarray(theta, dtype=float)))
    c = to_ggah_cosmology(theta)
    emu.set_cosmos(Omegab=float(c.Omega_b),
                   Omegac=float(c.Omega_cdm),
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
