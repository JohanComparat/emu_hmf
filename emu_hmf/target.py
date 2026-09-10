r"""What is being fitted: the CSST mass function, in this package's variance convention.

**The target** is CSSTemu's emulated :math:`\dd n/\dd\ln M`, a Gaussian process
trained on the CSST suite, so it carries the simulations' calibration rather
than a fit to them.  It is numpy and not differentiable, and is evaluated
offline.

**The variance** comes from ``ggah_mod``: the cold field against
:math:`\bar\rho_{cb}`, which is the :math:`\sigma(M)` the recalibrated fit is
evaluated with.  Fitting :math:`f(\sigma)` against one variance and evaluating
it with another shifts the answer by the difference between the two
conventions.

The shipped training set was built on CLASS rather than on a network spectrum,
because :mod:`emu_pk` does not cover this box; :mod:`emu_hmf.generate` gives the
figure.  :func:`sigma_chain` defaults to :mod:`emu_pk` as a convenience for
callers, so the weights and the accuracies quoted for them do not depend on
which spectrum emulator is installed.

Which mass definition
---------------------

CSSTemu offers ``RockstarM200m``, ``FoFM200c`` and ``RockstarMvir``.  The only
:math:`200{\rm c}` on offer is a friends-of-friends mass, and a FoF catalogue
and a spherical-overdensity multiplicity function count different objects.

The default is therefore ``RockstarM200m``: a spherical-overdensity mass at the
definition ``tinker08`` was calibrated in.  ``ggah_mod`` reaches
:math:`200{\rm c}` afterwards through the published :math:`\log\Delta`
interpolation.  ``FoFM200c`` is selectable, and carries a change of halo finder
along with the definition.
"""

from __future__ import annotations

import numpy as np

from . import box

__all__ = ["MASSDEFS", "DEFAULT_MASSDEF", "Z_TRAINED", "M_TRUSTED",
           "NU_TRUSTED", "NU_COVERED", "nu_covered", "DELTA_C", "FIDUCIAL",
           "OMEGA_K_COST", "OMEGA_K_CROSSOVER", "OMEGA_K_MEASURED_TO",
           "OMEGA_K_MEASURED_CROSSING", "crossover_is_measured",
           "NU_ORDERING_COST",
           "GGAH_MIN_VERSION",
           "csst_dndlnM", "csst_tinker08",
           "set_cosmology", "sigma_chain", "to_ggah_cosmology",
           "theta_from_cosmology"]

#: Spherical-collapse threshold, :math:`\delta_c`, in the
#: :math:`\nu = \delta_c/\sigma` that :data:`NU_TRUSTED` is stated in.
#:
#: The Einstein--de Sitter value, held fixed rather than made cosmology
#: dependent, because that is the convention Tinker08 was calibrated in.  A
#: cosmology-dependent :math:`\delta_c` would move the peak-height cut with
#: the cosmology and make the training range mean a slightly different thing at
#: every design point.
DELTA_C = 1.686

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


#: What the training set *actually* spans at each trained redshift.
#:
#: :data:`NU_TRUSTED` states the peak heights the emulator can be trusted over,
#: but the training set is bounded in mass as well (:data:`M_TRUSTED`), and the
#: two cuts do not commute with redshift.  Growth pushes :math:`\sigma` down, so
#: a fixed mass is a higher peak at higher :math:`z`: the same
#: :math:`10^{12}\,M_\odot/h` that sits at :math:`\nu = 0.5` today sits at
#: :math:`\nu = 1.4` at :math:`z = 3`.  The low-:math:`\nu` half of the band is
#: therefore simply absent above :math:`z \simeq 0.25`, and a correction
#: evaluated there is an extrapolation whatever :data:`NU_TRUSTED` says.
#:
#: Measured on the shipped training set, as ``z -> (nu_lo, nu_hi)``:
NU_COVERED = {
    0.00: (0.50, 2.97), 0.10: (0.50, 2.99), 0.25: (0.50, 3.00),
    0.50: (0.55, 3.00), 0.80: (0.65, 3.00), 1.00: (0.71, 3.00),
    1.25: (0.80, 3.00), 1.50: (0.88, 3.00), 1.75: (0.97, 3.00),
    2.00: (1.05, 3.00), 2.50: (1.23, 3.00), 3.00: (1.40, 3.00),
}


def nu_covered(z):
    r"""``(nu_lo, nu_hi)`` the training set spans at redshift ``z``.

    Linear in ``z`` between the trained redshifts, clamped outside them.  Use
    it to ask whether a :math:`(\sigma, z)` was actually fitted, rather than
    only whether it is inside :data:`NU_TRUSTED` --- above :math:`z \simeq
    0.25` those are different questions.
    """
    zs = np.array(sorted(NU_COVERED))
    lo = np.interp(z, zs, [NU_COVERED[k][0] for k in zs])
    hi = np.interp(z, zs, [NU_COVERED[k][1] for k in zs])
    return float(lo), float(hi)


#: What curvature costs, as :math:`\max|\Delta\ln f|` per unit
#: :math:`|\Omega_k|`, one coefficient per weights file.
#:
#: :data:`emu_hmf.box.PARAMS` has no curvature axis, so
#: :func:`theta_from_cosmology` passes a curved cosmology through as the flat
#: one carrying the same eight numbers.  Closing that gap would need curved
#: simulations rather than a longer fit, so this package measures the cost
#: instead.
#:
#: Measured by restoring the :math:`\Omega_k(1+z)^2` term the emulator's own
#: :math:`E(z)` carries, which ``set_cosmos`` sets to zero, and re-evaluating
#: its Castro+23 baseline at fixed :math:`\sigma`.  On a 40-point design over
#: every trained redshift and :math:`10^{12}` to :math:`10^{14}\,M_\odot/h`:
#:
#: ==========  ===========  =================  ===========
#: weights     coefficient  residual at 0.002  crossover
#: ==========  ===========  =================  ===========
#: ``200m``    0.2242       0.00520            0.3009
#: ``vir``     0.8911       0.00571            0.1162
#: ==========  ===========  =================  ===========
#:
#: Virial is 3.97 times as sensitive, so the two do not pool.
#: :data:`OMEGA_K_CROSSOVER` is the more useful column: below it the
#: recalibration beats the ``tinker08`` it replaces, above it the carrier wins.
#: At the Planck and BAO bound of 0.002 the 200m residual moves from 0.00518 to
#: 0.00520 against 0.06766 for ``tinker08``, a factor of thirteen.
#:
#: **The limit, which belongs with the number.**  This measures the carrier's
#: response.  The CSST suite contains no curved simulations, so the response of
#: the Gaussian-process residual is unmeasured.  The assumption is that
#: curvature reaches a multiplicity function at fixed :math:`\sigma` through the
#: growth history in :math:`\Omega_m(z)`, and that the target's own model is
#: parameterised in that quantity.
#:
#: The Gaussian-process ratio is curvature-blind: perturbing the full emulated
#: ``dn/dlnM`` rather than the carrier alone changes the answer by
#: :math:`3\times10^{-6}` in :math:`\ln f` at :math:`|\Omega_k| = 0.3`.
OMEGA_K_COST = {"200m": 0.2242, "vir": 0.8911}

#: The largest :math:`|\Omega_k|` :data:`OMEGA_K_COST` was measured at, per
#: file.  Beyond it the linear law is an extrapolation --- a conservative one,
#: since the true coefficient falls, but an extrapolation.
#:
#: A dict rather than a scalar, for the reason :data:`OMEGA_K_COST` is one: a
#: pooled number would be a claim about a file it was not measured on.
OMEGA_K_MEASURED_TO = {"200m": 0.45, "vir": 0.45}

#: Where the crossing was actually *observed*, per file, or ``None``.
#:
#: A number rather than a flag, because a flag says the crossover might be off
#: and a number says by how much.
#:
#: Carried at the archive's own precision rather than rounded like
#: :data:`OMEGA_K_COST`.  Those are *quoted* figures; this one is *read*, and
#: two packages comparing a read value have no reason to leave slack for a
#: rounding neither of them performs.
#:
#: The sweep brackets both crossings.  At ``200m`` the correction still beats
#: ``tinker08`` at :math:`|\Omega_k| = 0.35` and no longer does at 0.40, placing
#: the crossing at 0.3715 against the linear law's 0.3009.  At ``vir`` the
#: brackets are 0.10 and 0.15, placing it at 0.1210 against 0.1162.  The law
#: lands early on both, by 19 and 4 per cent, and by more where it reaches
#: further, as a falling coefficient does.
OMEGA_K_MEASURED_CROSSING = {"200m": 0.371517, "vir": 0.120979}


def crossover_is_measured(massdef: str) -> bool:
    """Whether :data:`OMEGA_K_CROSSOVER` was bracketed rather than extrapolated.

    Derived from :data:`OMEGA_K_MEASURED_CROSSING` rather than stored, so there
    is no second place for one fact to disagree with itself.
    """
    return OMEGA_K_MEASURED_CROSSING[massdef] is not None



#: Where curvature makes the recalibration no better than its own carrier.
#:
#: :math:`|\Omega_k|` at which :data:`OMEGA_K_COST`, added in quadrature to a
#: file's held-out residual, reaches the ``tinker08`` baseline that file was
#: measured against.  Derived from the two numbers each weights file already
#: records, so it cannot drift from them.
#:
#: 0.3009 at 200m is twice :mod:`emu_pk`'s box edge and outside any prior in
#: use.  0.1162 at virial is inside one, so the two are separate entries.
OMEGA_K_CROSSOVER = {"200m": 0.3009, "vir": 0.1162}


#: The direction of the approximation, asserted at import.
#:
#: Quoting a falling coefficient as a linear law overestimates the cost, so a
#: crossover computed from it lands early and the
#: threshold refuses slightly too soon.  Early costs a little reach.  Late would
#: mean recommending a correction that is already worse than the carrier it
#: replaces, and every number involved would still look entirely reasonable ---
#: so nothing downstream would catch it.
#:
#: So wherever a crossing has been observed, the linear crossover must sit at or
#: below it.  This asserts the *direction* and not the values, so a regeneration
#: that moves the numbers stays legal and one that inverts the inequality does
#: not.  ``ggah_mod`` carries the same assertion against the same quantities;
#: this is the copy at the end that produces them.
for _md, _observed in OMEGA_K_MEASURED_CROSSING.items():
    if _observed is not None:
        assert OMEGA_K_CROSSOVER[_md] <= _observed, (
            f"the linear crossover for {_md} ({OMEGA_K_CROSSOVER[_md]}) is "
            f"above the measured crossing ({_observed}), so the law now "
            f"underestimates the cost and the threshold would refuse late "
            f"rather than early.  Late is the failure that matters and it "
            f"looks reasonable from every other angle.")
del _md, _observed


#: What the neutrino mass ordering costs, as ``sum_mnu -> (max, rms)``
#: :math:`|\Delta\ln f|` against the degenerate split the weights were fitted
#: under.
#:
#: **The correction itself cannot move.**  ``ggah_mod`` builds
#: :attr:`Omega_nu_matter` from :math:`\Sigma m_\nu/3` in closed form, so the
#: whole matter budget --- :attr:`Omega_cb`, :attr:`Omega_cdm`, :attr:`f_nu`,
#: :attr:`rho_cold` --- is bit-for-bit invariant under the ordering.
#: :func:`theta_from_cosmology` therefore returns *identical* numbers for a
#: normal and a degenerate cosmology at the same sum, and :math:`g(\theta, z)`
#: does not move at all.  What moves is the :math:`\sigma(M)` the caller passes
#: in, and with it the target the shipped fit was made against.
#:
#: Measured with CLASS, which takes the three masses, over the trained redshifts
#: and the peak heights the fit covers.  Normal ordering against degenerate;
#: it has no solution below 0.058993 eV, which is where the table starts:
#:
#: ==============  ==============  ==============
#: Sum m_nu [eV]   max shift       rms shift
#: ==============  ==============  ==============
#: 0.0590          1.15e-04        3.54e-05
#: 0.0600          1.13e-04        3.48e-05
#: 0.0700          9.79e-05        3.04e-05
#: 0.1000          6.71e-05        2.15e-05
#: 0.1500          3.90e-05        1.31e-05
#: 0.2000          2.53e-05        8.47e-06
#: 0.3000          1.23e-05        4.34e-06
#: ==============  ==============  ==============
#:
#: Largest where the orderings differ most, at the floor, and falling
#: monotonically as the split closes on degenerate.  The worst entry is 2.2 per
#: cent of the shipped held-out residual and the worst rms is 0.7 per cent, so
#: added in quadrature the residual is unchanged at both figures it is published
#: to.  **That is why a caller's ordering is accepted rather than refused**: the
#: threshold was fixed before the measurement, and this clears it by two orders
#: of magnitude.
#:
#: A caller whose :math:`\sigma(M)` came from a normal-ordered Boltzmann solve
#: is therefore *more* accurate than the training set, not less, and this table
#: bounds the mismatch.
NU_ORDERING_COST = {
    0.0590: (1.15e-04, 3.54e-05), 0.0595: (1.14e-04, 3.51e-05),
    0.0600: (1.13e-04, 3.48e-05), 0.0650: (1.05e-04, 3.25e-05),
    0.0700: (9.79e-05, 3.04e-05), 0.0800: (8.57e-05, 2.71e-05),
    0.1000: (6.71e-05, 2.15e-05), 0.1250: (5.05e-05, 1.68e-05),
    0.1500: (3.90e-05, 1.31e-05), 0.2000: (2.53e-05, 8.47e-06),
    0.2500: (1.69e-05, 5.94e-06), 0.3000: (1.23e-05, 4.34e-06),
}


#: The ``ggah_mod`` release this package's conversions need, for the error
#: message.  It is not what is *checked* --- see ``_require_ggah``.
GGAH_MIN_VERSION = "0.7.0"


def _require_ggah(Cosmology) -> None:
    """Refuse a ``ggah_mod`` too old for the conventions this module speaks.

    **Probed, not read off a version number**, for two reasons that point the
    same way.  Both changes this depends on --- :attr:`Omega_nu_matter` and the
    ``nu_hierarchy`` field --- landed *after* ``ggah_mod`` tagged 0.6.0 and
    before it tagged anything else, so a floor of ``>= 0.6.0`` is satisfied by a
    tree that has neither and a floor of ``>= 0.7.0`` is satisfied by nothing
    yet.  And every environment in this family installs ``ggah_mod`` editable
    from a checkout, where the version string says what was last tagged rather
    than what the tree contains.

    So this asks the class what it has, which is the same rule
    ``ggah_mod.halos._cemulator_compat`` follows one package over: probe the
    failure rather than the version, and a fixed upstream stops being patched.
    """
    #: Each missing name explains *itself*.  A message that recited both
    #: whenever either was absent would send the reader hunting for a second
    #: problem that is not there --- which is the same failure as a refusal
    #: naming only the first offender, one level down.
    why = {
        "Omega_nu_matter":
            "Omega_nu_matter is the density Omega_cb subtracts, and adding any "
            "other one puts 1e-4 of Omega_cb -- and of rho_cold, and so of "
            "sigma(M) -- into the variance this correction is defined against",
        "nu_hierarchy":
            "nu_hierarchy is what pins the degenerate split the shipped weights "
            "were fitted under, without which a fifth of the CSST box has no "
            "solution at all",
    }
    missing = []
    if not hasattr(Cosmology, "Omega_nu_matter"):
        missing.append("Omega_nu_matter")
    if "nu_hierarchy" not in getattr(Cosmology, "__dataclass_fields__", {}):
        missing.append("nu_hierarchy")
    if missing:
        raise ImportError(
            f"this ggah_mod is too old for emu_hmf's conversions: its "
            f"Cosmology has no {' and no '.join(missing)}.  "
            + "; ".join(why[m] for m in missing)
            + f".  That cannot be worked around here.  Install ggah_mod >= "
              f"{GGAH_MIN_VERSION}.")


def to_ggah_cosmology(theta):
    r"""CSST's eight -> a ``ggah_mod`` ``Cosmology``.

    Two of the conversions are conventional and stated in the box's own
    documentation: ``H0`` is in km/s/Mpc where ``ggah_mod`` wants ``h``, and
    ``A`` is :math:`10^{9}A_s` where ``ggah_mod`` wants :math:`\ln(10^{10}A_s)`.
    The third is not stated anywhere and is the one that costs something.

    CSST's ``Omegam`` --- the symbol ``param_limits`` bounds --- is the **cold**
    density, :math:`\Omega_b + \Omega_{cdm}`, with the massive neutrinos
    excluded; CSSTemu carries the total separately as ``Cosmo.OmegaM``.
    ``ggah_mod``'s :attr:`Omega_m` is the total.  So the two agree on
    :attr:`Omega_cb`, not on :attr:`Omega_m`, and the neutrino density has to be
    added on the way in.

    It is added by *asking ``ggah_mod``* rather than by dividing
    :math:`\Sigma m_\nu` by a constant here: the neutrino density is a
    :math:`\Sigma m_\nu`-and-:math:`h` quantity that does not depend on
    :attr:`Omega_m`, so one throwaway construction reads it off in whatever
    convention the package actually uses, and the second construction is exact
    by that package's own definition instead of by a constant repeated in two
    repositories.

    **And the property added must be the property the inverse subtracts.**  That
    is the whole rule, and it is what broke.  This function added
    :attr:`Omega_nu`, the 93.14 eV convention; ``ggah_mod`` then redefined
    :attr:`Omega_cb` to subtract :attr:`Omega_nu_matter` instead --- the
    matter-like part of the Fermi-Dirac density, which sits
    :math:`4.57\times10^{-3}` above it --- because subtracting the convention
    while ``hubble_e`` added the integral left the model carrying too much total
    matter.  Naming two different neutrino densities in the two directions is
    the failure, not the value of either: it cost
    :math:`1.0\times10^{-4}` of :attr:`Omega_cb` at
    :math:`\Sigma m_\nu = 0.3`, straight into :attr:`rho_cold` and so into the
    variance the correction is fitted against, and it pushed the
    :math:`(\Omega_{cb} = 0.24,\ \Sigma m_\nu = 0.3)` corner back out of the
    box it was sampled inside.

    ``nu_hierarchy`` is pinned to ``"degenerate"`` for a reason of the same
    kind.  ``ggah_mod``'s own default is ``"normal"``, three unequal masses from
    the oscillation splittings, which has **no solution below 0.058993 eV** ---
    so an unpinned construction refuses 393 of this package's own 2000 design
    points, a fifth of the box, while the fiducial 0.06 eV clears the floor by
    0.001 and every smoke test passes.  ``"degenerate"`` is also what the
    shipped weights were fitted under: it is what ``ClassPk`` reaches through
    ``deg_ncdm = 3``, which is the only behaviour that existed when the training
    set was generated.  Pinning it makes that a statement rather than a default
    inherited from another package.

    A caller's own ``nu_hierarchy`` is not ignored by any of this.  It shapes
    the :math:`\sigma(M)` they pass in, which is where the ordering belongs;
    :math:`\theta` cannot carry it and does not need to, because the matter
    budget is invariant under the split.
    """
    from ggah_mod.cosmology import Cosmology

    _require_ggah(Cosmology)
    d = dict(zip(box.PARAMS, np.asarray(theta, dtype=float)))
    kw = dict(Omega_b=d["Omegab"], h=d["H0"] / 100.0, n_s=d["ns"],
              ln10A_s=float(np.log(10.0 * d["A"])),
              sum_mnu=d["mnu"], w0=d["w"], wa=d["wa"],
              nu_hierarchy="degenerate")
    probe = Cosmology.create(Omega_m=d["Omegam"], **kw)
    return Cosmology.create(
        Omega_m=d["Omegam"] + float(probe.Omega_nu_matter), **kw)


def theta_from_cosmology(cosmo):
    r"""A ``ggah_mod`` ``Cosmology`` -> CSST's eight, in :data:`emu_hmf.box.PARAMS` order.

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

    **Two of the caller's parameters are dropped, and both are dropped
    knowingly.**  :attr:`Omega_k` and ``nu_hierarchy`` have no column here,
    because :data:`emu_hmf.box.PARAMS` is CSSTemu's box and that box is flat with one
    neutrino species.  Neither can be added by retraining: the target is a
    *simulation suite*, so a curvature axis would need curved simulations rather
    than a longer fit.

    The ordering costs nothing to drop, and that is provable rather than
    hopeful: ``ggah_mod`` builds :attr:`Omega_nu_matter` from
    :math:`\Sigma m_\nu/3` in closed form, so every density this reads is
    bit-for-bit invariant under the split and the eight numbers returned are
    *identical* for a normal and a degenerate cosmology.  What the ordering
    changes is the caller's :math:`\sigma(M)`, which is an input and is theirs.
    :data:`NU_ORDERING_COST` bounds the mismatch against the training set.

    Curvature does cost something, and :data:`OMEGA_K_COST` says how much.  **No
    warning is raised here**, deliberately: this function is documented as
    traceable and never asks for a concrete value, so a check would have to be
    skipped under ``jit`` exactly where a forward model lives.  The policy
    belongs one layer up, where the cosmology is concrete and the mass
    definition is known --- ``ggah_mod.halos.mass_function`` consumes these
    constants and refuses past :data:`OMEGA_K_CROSSOVER`.  Exporting a measured
    number and letting the caller act on it once beats two half-checks that
    disagree.
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
    r"""Hand CSSTemu a cosmology, in the dialect ``set_cosmos`` speaks.

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
    r"""CSSTemu's *own* Tinker08, at an explicit :math:`\Delta_{\rm m}`.

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

    The shim is not optional and not a version check.  Two of CSSTemu's
    methods assign a size-1 array into a scalar slot, which numpy 2 refuses, so
    the mass function raises before returning anything.  ``ggah_mod`` carries
    the patch already --- *probed* by calling the method and catching the error,
    so a fixed upstream stops being patched --- and it has to be applied after a
    cosmology is set, because that is what makes the probe reach the failure.
    """
    from ggah_mod.halos._cemulator_compat import ensure_cemulator_works
    from CEmulator.Emulator import HMF_CEmulator
    emu = HMF_CEmulator()
    set_cosmology(emu, FIDUCIAL if theta is None else theta)
    ensure_cemulator_works(emu)
    if theta is not None:
        set_cosmology(emu, theta)
    return emu


#: A Planck-like point in the middle of the CSST box, in :data:`emu_hmf.box.PARAMS`
#: order.  Public because it is the cosmology every worked example, figure and
#: quoted number in this package is evaluated at, and one definition of it is
#: better than five.
FIDUCIAL = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.06])


def sigma_chain(theta, z, m, pk=None):
    r"""``(sigma, dlnsigma_dlnM, rho_cb)`` in ``ggah_mod``'s convention.

    The cold field against :math:`\bar\rho_{cb}`, and the logarithmic derivative
    by automatic differentiation of the same integral rather than by
    differencing it -- both because that is what the recalibrated fit will be
    evaluated with, and because a finite difference of a quadrature is how
    percent-level noise gets into a mass function.

    **A convenience, not the generation path.**  The default backend is
    :mod:`emu_pk`, which is fast and differentiable but does not cover this box:
    :math:`\omega_b = \Omega_b h^2` leaves its bounds over 30 per cent of the
    CSST design (see :mod:`emu_hmf.box`), and it refuses there rather than
    extrapolating.  Pass ``pk=make_pk("class")`` for a cosmology it turns down,
    which is what :mod:`emu_hmf.generate` uses throughout.

    The numbers this returns moved with :mod:`emu_pk` 2.0.0, whose weights are
    fitted on a wider box than 1.0.0's.  Nothing shipped in this package moved
    with them: the training set was built with CLASS, so the weights and every
    accuracy quoted for them are independent of which spectrum emulator is
    installed.
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
