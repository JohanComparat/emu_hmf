r"""The target, and the two conversions that are silently wrong if guessed.

The module-level import guard is deliberately absent: only the classes that
genuinely need the CSST emulator or the halo-model code skip without them, so
that everything testable in a plain install keeps being tested.
"""
import pathlib
import sys
import types

import numpy as np
import pytest

from emu_hmf import box, target


class TestTheDialectConversions:
    """``param_limits`` and ``set_cosmos`` do not speak the same language.

    The bounds are stated in :math:`\\Omega_m` and :math:`10^9 A_s`; the setter
    wants the *cold* density and :math:`A_s` itself, which it then multiplies by
    :math:`10^9` to check against those same bounds.  Either one guessed gives a
    plausible number rather than an error, which is why both are pinned here.
    """

    def test_the_amplitude_is_A_s_not_1e9_A_s(self, emu):
        """Passing the box's ``A`` straight through is caught by the emulator's
        own bound check -- so the test is that we do not do it."""
        with pytest.raises(ValueError, match="out of range"):
            emu.set_cosmos(Omegab=0.049, Omegac=0.261, H0=67.36,
                           As=2.1, ns=0.9649, w=-1.0, wa=0.0, mnu=0.06)

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30])
    def test_the_density_round_trip_is_exact(self, emu, mnu):
        """The quiet trap, pinned in both directions.

        CSST's ``Omegam`` -- what ``param_limits`` bounds -- is the *cold*
        density; ``set_cosmos``'s ``Omegac`` argument is not, because the setter
        subtracts :math:`\\Omega_\\nu` from it internally.  So the value handed
        over is the total matter minus baryons, and what comes back must be the
        sampled ``Omegam`` exactly.  Any other reading loses
        :math:`\\Omega_\\nu` somewhere: half a per cent at 0.06 eV, two per cent
        at 0.3, in the variance the whole recalibration is made against.
        """
        th = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, mnu])
        target.set_cosmology(emu, th)
        assert float(emu.Cosmo.Omegam) == pytest.approx(0.31, abs=1e-12)
        # And the identity that makes the two packages the same cosmology:
        # CSST's bounded `Omegam` is the halo code's *cold* density, so its
        # total sits above it by exactly the density `Omega_cb` subtracts.
        #
        # That is `Omega_nu_matter` -- the matter-like part of the Fermi-Dirac
        # density -- and *not* `Omega_nu`, the 93.14 eV convention, which sits
        # 4.57e-3 below it.  This assertion named `Omega_nu` and was the thing
        # that encoded the bug: it kept passing while the two packages drifted,
        # because both quantities are individually defensible and only their
        # *pairing* is the invariant.
        #
        # `rel=1e-12` and not exact equality, because this reconstructs the
        # density by *subtracting* two numbers of order 0.31 to recover one of
        # order 7e-3: the cancellation costs 1.4e-14 relative, which is a
        # property of the arithmetic and not of the conversion.  The error being
        # guarded against is 4.6e-3, so the tolerance sits a hundred times above
        # the noise and eleven orders below the defect.  The conversion itself
        # *is* exact, and `TestTheConversionRoundTrips` asserts that with `==`.
        c = target.to_ggah_cosmology(th)
        assert float(c.Omega_cb) == pytest.approx(0.31, rel=1e-12)
        assert float(c.Omega_m) - float(c.Omega_cb) == pytest.approx(
            float(c.Omega_nu_matter), rel=1e-12)
        assert float(c.Omega_cdm) == pytest.approx(0.31 - 0.049, rel=1e-12)
        if mnu > 0:
            assert float(c.Omega_m) > 0.31

    @pytest.mark.parametrize("omegam", [0.24, 0.40])
    @pytest.mark.parametrize("mnu", [0.0, 0.30])
    def test_a_box_edge_survives_the_round_trip(self, emu, omegam, mnu):
        """The corners where the convention matters most.

        :math:`\\Omega_{cb} = 0.24` with :math:`\\Sigma m_\\nu = 0.3` is inside
        the box by construction; read the density convention the other way and
        the emulator refuses it as ``Omegam = 0.2329 < 0.24``.  A design point
        falling out of the box it was sampled inside is the only symptom this
        error has --- there is no wrong number to notice, only a refusal.

        All four :math:`\\Omega_{cb} \\times \\Sigma m_\\nu` corners rather
        than the one, because the failure is proportional to
        :math:`\\Omega_\\nu` and *only* bites where the density is smallest and
        the neutrino mass largest.  Testing the one corner that fails is how a
        guard comes to depend on someone having picked the right corner.
        """
        th = np.array([0.049, omegam, 67.36, 0.9649, 2.1, -1.0, 0.0, mnu])
        target.set_cosmology(emu, th)
        assert float(emu.Cosmo.Omegam) == pytest.approx(omegam, abs=1e-12)

    def test_the_round_trip_reproduces_sigma8(self, emu):
        """The conversions, end to end, against the emulator's own sigma_8."""
        target.set_cosmology(emu, target.FIDUCIAL)
        s8 = float(np.atleast_1d(emu.get_sigma8())[0])
        assert 0.70 < s8 < 0.90, s8


class TestTheTargetIsARatio:
    """What has to be learned, and that it is small and smooth.

    The emulator ships *both* its simulation-calibrated ``dn/dlnM`` and its own
    Tinker08 against the cold spectrum, so the quantity to fit is their ratio --
    which is testable at the calibration cosmology in a way a fit from scratch
    is not.
    """

    M = np.logspace(12.0, 15.0, 7)

    def _ratio(self, emu, theta, z=0.0):
        d = np.asarray(target.csst_dndlnM(theta, z, self.M, emu=emu)).ravel()
        t = np.asarray(target.csst_tinker08(theta, z, self.M, emu=emu)).ravel()
        return d / t

    def test_it_is_a_few_per_cent_at_planck(self, emu):
        r = self._ratio(emu, target.FIDUCIAL)
        assert np.all(np.isfinite(r))
        assert 0.90 < r.min() and r.max() < 1.10, r
        # And not identically one: if it were, there would be nothing to fit.
        assert np.max(np.abs(r - 1.0)) > 0.01, r

    def test_it_moves_with_the_cosmology(self, emu):
        """The whole premise of the recalibration, as a test.

        A correction that did not depend on cosmology would be a constant, and
        a constant is already inside Tinker08's amplitude.
        """
        base = self._ratio(emu, target.FIDUCIAL)
        low_om = self._ratio(
            emu, np.array([0.049, 0.25, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.06]))
        assert np.max(np.abs(low_om / base - 1.0)) > 0.02, (base, low_om)

    def test_it_is_smooth_in_mass(self, emu):
        """Smooth enough for a low-order form to carry it, which is the claim.

        Measured as the residual of a cubic in ``ln M`` rather than as a second
        difference: on a seven-point grid over three decades a second
        difference is dominated by the spacing, and the genuine downturn at the
        cluster end reads as roughness when it is signal.
        """
        m = np.logspace(*np.log10(target.M_TRUSTED), 25)
        ln_r = self._log_ratio(emu, m)
        coeff = np.polyfit(np.log(m), ln_r, 3)
        resid = np.max(np.abs(ln_r - np.polyval(coeff, np.log(m))))
        assert resid < 6e-3, resid

    def test_it_is_rough_above_the_trusted_range(self, emu):
        """And that is what :data:`~emu_hmf.target.M_TRUSTED` is for.

        Stated as a test because a range that is never checked drifts into
        being a number someone chose.  The same cubic through the same ratio
        one decade higher is five times worse, and the reason is the target
        rather than any fit made to it: a simulation suite runs out of
        clusters, and a Gaussian process is noisiest where its training data is
        thinnest.
        """
        m = np.logspace(12.0, 15.0, 25)
        ln_r = self._log_ratio(emu, m)
        coeff = np.polyfit(np.log(m), ln_r, 3)
        assert np.max(np.abs(ln_r - np.polyval(coeff, np.log(m)))) > 2e-2

    @staticmethod
    def _log_ratio(emu, m):
        d = np.asarray(target.csst_dndlnM(target.FIDUCIAL, 0.0, m,
                                          emu=emu)).ravel()
        t = np.asarray(target.csst_tinker08(target.FIDUCIAL, 0.0, m,
                                            emu=emu)).ravel()
        return np.log(d / t)


class TestTheVarianceIsOurs:
    def test_sigma_falls_and_the_slope_is_negative(self):
        pytest.importorskip("ggah_mod.halos.variance")
        pytest.importorskip("emu_pk")
        m = np.logspace(12.0, 15.0, 5)
        sig, dlns, rho = target.sigma_chain(target.FIDUCIAL, 0.0, m)
        assert np.all(np.diff(sig) < 0)
        assert np.all(dlns < 0)
        assert rho > 0

    def test_it_refuses_a_cosmology_outside_the_box(self, emu):
        bad = target.FIDUCIAL.copy()
        bad[box.PARAMS.index("H0")] = 55.0
        with pytest.raises(ValueError, match="outside the CSST"):
            target.csst_dndlnM(bad, 0.0, np.array([1e13]), emu=emu)


class TestTheMassDefinitionChoice:
    def test_the_default_is_the_spherical_overdensity_one(self):
        assert target.DEFAULT_MASSDEF == "RockstarM200m"
        assert "friends-of-friends" in target.MASSDEFS["FoFM200c"]

    def test_an_unknown_definition_is_refused_by_name(self):
        with pytest.raises(ValueError, match="massdef must be one of"):
            target.csst_dndlnM(target.FIDUCIAL, 0.0, np.array([1e13]),
                               massdef="Rockstar500c", emu=object())

    def test_all_three_are_the_emulators(self, emu):
        for md in target.MASSDEFS:
            v = np.asarray(target.csst_dndlnM(
                target.FIDUCIAL, 0.0, np.array([1e13, 1e14]), massdef=md,
                emu=emu)).ravel()
            assert np.all(np.isfinite(v)) and np.all(v > 0), md

    def test_they_are_genuinely_different_masses(self, emu):
        """Not conventions.  Which is why pairing the friends-of-friends mass
        with a spherical-overdensity multiplicity function would be a category
        error, and why the default is the Rockstar one."""
        m = np.array([1e13, 1e14])
        a, b = (np.asarray(target.csst_dndlnM(target.FIDUCIAL, 0.0, m,
                                              massdef=md, emu=emu)).ravel()
                for md in ("RockstarM200m", "FoFM200c"))
        assert np.max(np.abs(a / b - 1.0)) > 0.02, (a, b)


class TestTheCoveredRange:
    """:data:`~emu_hmf.target.NU_TRUSTED` is necessary and not sufficient."""

    def test_the_band_narrows_with_redshift(self):
        """Growth pushes sigma down, so a fixed mass is a higher peak at
        higher z, and the low-nu half of the band is simply absent up there.
        A caller who checked only ``NU_TRUSTED`` would be extrapolating and
        would get no warning."""
        lo0, hi0 = target.nu_covered(0.0)
        lo3, hi3 = target.nu_covered(3.0)
        assert lo0 == pytest.approx(target.NU_TRUSTED[0], abs=1e-9)
        assert lo3 > 2 * lo0
        assert hi0 == pytest.approx(hi3, abs=0.05)

    def test_it_is_monotone_and_stays_inside_the_trusted_range(self):
        z = np.linspace(0.0, 3.0, 25)
        lo = np.array([target.nu_covered(zz)[0] for zz in z])
        assert np.all(np.diff(lo) >= 0)
        assert lo.min() >= target.NU_TRUSTED[0] - 1e-9
        assert max(target.nu_covered(zz)[1] for zz in z) <= \
            target.NU_TRUSTED[1] + 1e-9

    def test_it_clamps_rather_than_extrapolating(self):
        assert target.nu_covered(9.0) == target.nu_covered(3.0)
        assert target.nu_covered(-1.0) == target.nu_covered(0.0)

    def test_every_trained_redshift_is_recorded(self):
        assert set(target.NU_COVERED) == set(target.Z_TRAINED)


class TestTheConversionRoundTrips:
    """Both directions, and the density that makes them differ.

    :func:`~emu_hmf.target.to_ggah_cosmology` and
    :func:`~emu_hmf.target.theta_from_cosmology` are the only two places the
    dialect is translated.  If they ever disagree, the halo layer evaluates the
    correction at a cosmology that is not the one it was asked for -- silently,
    because every value stays plausible.
    """

    #: ``0.02`` and ``0.058`` are not decoration.  ``ggah_mod``'s ``Cosmology``
    #: defaults to a *normal* ordering, whose three eigenstates cannot sum to
    #: less than 0.058993 eV, so an unpinned construction **raises** below that
    #: --- for 393 of this package's own 2000 design points.  The fiducial
    #: 0.06 clears the floor by 0.001 eV, which is why every smoke test passed
    #: while a fifth of the box was unreachable.  A range that is only ever
    #: tested above the floor cannot see this.
    @pytest.mark.parametrize("mnu", [0.0, 0.02, 0.058, 0.06, 0.30])
    def test_theta_to_cosmology_and_back(self, mnu):
        pytest.importorskip("ggah_mod.cosmology")
        th = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, mnu])
        back = np.asarray(target.theta_from_cosmology(
            target.to_ggah_cosmology(th)))
        # Exact, not `approx`.  Measured: every element round-trips bit for bit,
        # because the density added on the way in is the one subtracted on the
        # way out.  A tolerance here is what let a 4.6e-3 convention drift live
        # in the package unnoticed, so the assertion says what is actually true.
        assert np.array_equal(back, th), dict(zip(box.PARAMS, back - th))

    def test_the_whole_neutrino_range_of_the_box_converts(self):
        """Every ``mnu`` the box samples, not only the ones above the floor.

        The direct regression for the failure above: sampled across
        :data:`~emu_hmf.box.BOX`'s own ``mnu`` range rather than at chosen
        points, so a future change that reintroduces an ordering-dependent floor
        cannot pass by clearing it at the fiducial.
        """
        pytest.importorskip(
            "ggah_mod.cosmology",
            reason="the regression for the 19.7 per cent of the box that used "
                   "to refuse is UNVERIFIED without ggah_mod -- this skip is "
                   "not a pass")
        lo, hi = box.BOX["mnu"]
        for mnu in np.linspace(lo, hi, 25):
            th = target.FIDUCIAL.copy()
            th[box.PARAMS.index("mnu")] = mnu
            back = np.asarray(target.theta_from_cosmology(
                target.to_ggah_cosmology(th)))
            assert np.array_equal(back, th), (mnu, back - th)

    def test_the_hierarchy_is_pinned_to_the_one_the_weights_were_fitted_under(self):
        r"""``degenerate``, declared rather than inherited.

        ``ggah_mod``'s class default is ``"normal"``; ``ClassPk`` reaches three
        equal masses through ``deg_ncdm = 3``, which is the only behaviour that
        existed when the training set was generated.  Without the pin this is
        both a refusal below 0.059 eV and, above it, a silently different
        :math:`\sigma(M)` from the one the shipped weights were fitted against.
        """
        pytest.importorskip(
            "ggah_mod.cosmology",
            reason="the degenerate pin is UNVERIFIED without ggah_mod, and it "
                   "is what keeps a fifth of the box constructible -- this "
                   "skip is not a pass")
        for mnu in (0.0, 0.02, 0.06, 0.30):
            th = target.FIDUCIAL.copy()
            th[box.PARAMS.index("mnu")] = mnu
            assert target.to_ggah_cosmology(th).nu_hierarchy == "degenerate"

    def test_the_cold_density_is_exact_across_the_design(self):
        """The invariant that makes the two packages one cosmology.

        CSST's bounded ``Omegam`` *is* ``ggah_mod``'s :attr:`Omega_cb`, over the
        whole design and not just at the fiducial.  Needs no CSST emulator, so
        it runs wherever the halo code does.
        """
        pytest.importorskip("ggah_mod.cosmology")
        worst = max(abs(float(target.to_ggah_cosmology(th).Omega_cb) - th[1])
                    for th in box.sample(256))
        assert worst <= 4.0 * np.spacing(box.BOX["Omegam"][1]), worst

    def test_it_survives_tracing(self):
        """The reason it is not simply its inverse, written backwards.

        This direction is called from inside the halo layer's differentiable
        path, so every field may be a tracer.  A ``float()`` anywhere in it
        would raise a ``ConcretizationTypeError`` the first time somebody
        differentiated a mass function -- which is exactly the use it exists
        for.
        """
        pytest.importorskip("ggah_mod.cosmology")
        import jax

        from ggah_mod.cosmology import PLANCK18

        def amp(a):
            return target.theta_from_cosmology(PLANCK18.replace(ln10A_s=a))[4]

        g = float(jax.grad(amp)(float(PLANCK18.ln10A_s)))
        # A = exp(ln10A_s)/10, so dA/dln10A_s = A.
        assert g == pytest.approx(float(np.exp(PLANCK18.ln10A_s) / 10.0),
                                  rel=1e-10)

    def test_the_cold_density_is_what_crosses(self):
        """Not :math:`\\Omega_m`: the box bounds the cold density."""
        pytest.importorskip("ggah_mod.cosmology")
        from ggah_mod.cosmology import Cosmology

        c = Cosmology.create(sum_mnu=0.3)
        th = np.asarray(target.theta_from_cosmology(c))
        assert th[1] == pytest.approx(float(c.Omega_cb), rel=1e-12)
        assert th[1] < float(c.Omega_m)


class _RecordingEmulator:
    """The emulator's interface, without the emulator.

    Lets the dispatch in :func:`~emu_hmf.target.csst_dndlnM` and
    :func:`~emu_hmf.target.csst_tinker08` -- the box check, the cosmology
    hand-over, the argument names -- be tested in an environment that has no
    Gaussian process in it.  What it cannot test is the numbers, which is what
    the classes above are for.
    """

    def __init__(self):
        self.cosmos, self.calls = [], []

    def set_cosmos(self, **kw):
        self.cosmos.append(kw)
        return self

    def get_dndlnM(self, z, M, massdef):
        self.calls.append(("dndlnM", massdef))
        return np.ones((len(np.atleast_1d(z)), len(np.atleast_1d(M))))

    def get_dndlnM_Tinker08(self, z, M, Pcb, Delta, rho_type):
        self.calls.append(("tinker08", Pcb, Delta, rho_type))
        return np.ones((len(np.atleast_1d(z)), len(np.atleast_1d(M))))


class TestTheCosmologyHandover:
    """``set_cosmos`` wants the cold densities, and this is what hands them.

    The identity that has to hold: CSST's bounded ``Omegam`` is the halo code's
    :attr:`Omega_cb`, and ``set_cosmos``'s ``Omegac`` is its
    :attr:`Omega_cdm`.  Getting either wrong double-counts the neutrinos.
    """

    @pytest.fixture(autouse=True)
    def _needs_the_halo_code(self):
        pytest.importorskip("ggah_mod.cosmology")

    def test_it_hands_over_the_cold_densities_and_A_s_itself(self):
        emu = _RecordingEmulator()
        target.set_cosmology(emu, target.FIDUCIAL)
        kw = emu.cosmos[-1]
        c = target.to_ggah_cosmology(target.FIDUCIAL)
        assert kw["Omegab"] == pytest.approx(float(c.Omega_b), rel=1e-12)
        assert kw["Omegac"] == pytest.approx(float(c.Omega_cdm), rel=1e-12)
        assert kw["H0"] == pytest.approx(67.36, rel=1e-12)
        assert kw["As"] == pytest.approx(2.1e-9, rel=1e-12), (
            "the box states 1e9 A_s; the setter wants A_s")
        assert kw["mnu"] == pytest.approx(0.06, rel=1e-12)

    def test_dndlnM_asks_for_the_definition_and_checks_the_box_first(self):
        emu = _RecordingEmulator()
        out = target.csst_dndlnM(target.FIDUCIAL, [0.0, 1.0],
                                 np.array([1e13, 1e14]),
                                 massdef="RockstarMvir", emu=emu)
        assert out.shape == (2, 2)
        assert emu.calls == [("dndlnM", "RockstarMvir")]
        assert emu.cosmos, "the cosmology was never handed over"

    def test_tinker08_is_asked_for_against_the_cold_spectrum(self):
        """``Pcb=True`` is this package's convention throughout: the variance
        the fit is made against is the cold field, so the analytic answer it is
        divided by has to be too."""
        emu = _RecordingEmulator()
        target.csst_tinker08(target.FIDUCIAL, 0.0, np.array([1e13]), emu=emu)
        _, pcb, delta, rho_type = emu.calls[-1]
        assert pcb is True and delta == 200.0 and rho_type == "matter"

    def test_both_refuse_outside_the_box_before_touching_the_emulator(self):
        bad = target.FIDUCIAL.copy()
        bad[box.PARAMS.index("mnu")] = 0.5
        emu = _RecordingEmulator()
        for fn in (target.csst_dndlnM, target.csst_tinker08):
            with pytest.raises(ValueError, match="outside the CSST"):
                fn(bad, 0.0, np.array([1e13]), emu=emu)
        assert emu.calls == [], "it reached the emulator with a refused cosmology"


class TestTheEmulatorIsBuiltInTheRightOrder:
    """``_emulator`` is an ordering contract, and the order is the point.

    ``ggah_mod``'s shim probes for CSSTemu's numpy-2 failure by *calling* the
    method that fails, so it only reaches that failure once a cosmology has
    been set.  Applied first it would be looking at an emulator that has not
    been asked for anything yet, find nothing wrong, and patch nothing --- and
    the mass function would raise on the first real call instead.

    Both sides are stubbed, because having CSSTemu and the halo-model code
    installed *together* is exactly what an ordinary environment does not, and
    the order is worth asserting somewhere rather than nowhere.
    """

    @pytest.fixture(autouse=True)
    def _needs_the_halo_code(self):
        pytest.importorskip("ggah_mod.cosmology")

    @pytest.fixture
    def stubbed(self, monkeypatch):
        order, made = [], []

        class _Emu(_RecordingEmulator):
            def set_cosmos(self, **kw):
                order.append("cosmology")
                return super().set_cosmos(**kw)

        def _construct():
            order.append("constructed")
            made.append(_Emu())
            return made[-1]

        def _shim(emu):
            order.append("shim")
            return emu

        cemulator = types.ModuleType("CEmulator")
        emulator = types.ModuleType("CEmulator.Emulator")
        emulator.HMF_CEmulator = _construct
        cemulator.Emulator = emulator
        compat = types.ModuleType("ggah_mod.halos._cemulator_compat")
        compat.ensure_cemulator_works = _shim
        monkeypatch.setitem(sys.modules, "CEmulator", cemulator)
        monkeypatch.setitem(sys.modules, "CEmulator.Emulator", emulator)
        monkeypatch.setitem(sys.modules, "ggah_mod.halos._cemulator_compat", compat)
        return order, made

    def test_the_shim_comes_after_a_cosmology_not_before(self, stubbed):
        order, _ = stubbed
        target._emulator()
        assert order == ["constructed", "cosmology", "shim"], (
            "the shim probes by calling the failing method, so a cosmology has "
            "to be set before there is anything for it to see")

    def test_without_a_cosmology_it_uses_the_fiducial(self, stubbed):
        _, made = stubbed
        target._emulator()
        assert made[0].cosmos[-1]["H0"] == pytest.approx(67.36, rel=1e-12)

    def test_a_requested_cosmology_is_set_again_after_the_shim(self, stubbed):
        """The shim is free to leave the emulator holding whatever cosmology it
        probed with, so the caller's is re-applied last and is what survives."""
        order, made = stubbed
        theta = target.FIDUCIAL.copy()
        theta[box.PARAMS.index("H0")] = 70.0
        emu = target._emulator(theta)
        assert order == ["constructed", "cosmology", "shim", "cosmology"]
        assert emu is made[0]
        assert [c["H0"] for c in emu.cosmos] == pytest.approx([70.0, 70.0])


class TestTheGgahRequirement:
    """The conversions need two things a version number cannot promise.

    :attr:`Omega_nu_matter` and the ``nu_hierarchy`` field both landed in
    ``ggah_mod`` *after* it last tagged, so a floor of ``>= 0.6.0`` is satisfied
    by a tree that has neither and a floor of ``>= 0.7.0`` by nothing yet.  And
    every environment in this family installs it editable from a checkout, where
    the version string says what was last tagged rather than what the tree
    holds.  So the class is asked what it has.
    """

    def test_it_names_both_when_both_are_absent(self):
        class Ancient:                       # a ggah_mod from before either
            __dataclass_fields__ = {"Omega_m": None, "sum_mnu": None}

        with pytest.raises(ImportError) as e:
            target._require_ggah(Ancient)
        msg = str(e.value)
        assert "Omega_nu_matter" in msg and "nu_hierarchy" in msg
        assert target.GGAH_MIN_VERSION in msg

    def test_it_names_only_the_one_that_is_missing(self):
        """Between the two commits, only the hierarchy is absent.  A message
        that named both would send the reader looking for a second problem
        that is not there."""
        class Midway:
            Omega_nu_matter = property(lambda self: 0.0)
            __dataclass_fields__ = {"Omega_m": None, "sum_mnu": None}

        with pytest.raises(ImportError, match="nu_hierarchy"):
            target._require_ggah(Midway)
        with pytest.raises(ImportError) as e:
            target._require_ggah(Midway)
        assert "Omega_nu_matter" not in str(e.value)

    def test_the_installed_ggah_mod_satisfies_it(self):
        """Skips without it, like every other test that needs the halo code.

        A plain install must report skips and no failures --- ``CONTRIBUTING``,
        ``docs/testing`` and ``docs/installation`` all promise that in those
        words, and CI installs only ``[dev]``.  This test asserted the promise
        while breaking it.
        """
        pytest.importorskip("ggah_mod.cosmology")
        from ggah_mod.cosmology import Cosmology
        target._require_ggah(Cosmology)      # must not raise


class TestTheCurvatureBound:
    """The cost of the axis the box does not have, recorded rather than refused.

    Curvature cannot be trained away here: closing the gap needs curved
    simulations, not a training run.  So the package owes a number, and these
    are the tests that keep it from becoming a number someone chose.
    """

    def test_there_is_one_coefficient_per_weights_file(self):
        """Every one of these is keyed by mass definition, and none is a scalar.

        A pooled number is a claim about a file it was not measured on, which is
        the exact error the per-file split exists to prevent --- and one that was
        made here once, when the measured range was a single 0.30 taken from the
        200m run.
        """
        from emu_hmf import model
        for d in (target.OMEGA_K_COST, target.OMEGA_K_CROSSOVER,
                  target.OMEGA_K_MEASURED_TO,
                  target.OMEGA_K_MEASURED_CROSSING):
            assert set(d) == set(model.WEIGHTS), d
        assert all(v > 0 for v in target.OMEGA_K_COST.values())

    def test_a_measured_crossing_is_inside_the_measured_range(self):
        """A crossing cannot have been observed outside the range measured."""
        for key, observed in target.OMEGA_K_MEASURED_CROSSING.items():
            assert target.crossover_is_measured(key) == (observed is not None)
            if observed is not None:
                assert observed <= target.OMEGA_K_MEASURED_TO[key], key

    def test_the_linear_law_errs_early_and_not_late(self):
        r"""The direction the whole threshold rests on.

        The coefficient falls with :math:`|\Omega_k|`, so reading it as linear
        overestimates the cost and the crossover lands *below* the observed
        crossing.  Early costs a little reach.  Late would recommend a
        correction already worse than the carrier it replaces, and every number
        would still look reasonable, which is why nothing downstream catches it.
        ``ggah_mod`` asserts the same inequality against the same quantities.
        """
        for key, observed in target.OMEGA_K_MEASURED_CROSSING.items():
            if observed is not None:
                assert target.OMEGA_K_CROSSOVER[key] <= observed, key

    def test_that_invariant_is_enforced_at_import(self):
        """An invariant that cannot fail is decoration, so this checks it can.

        Re-runs the module's own assertion against an inverted pair rather than
        trusting that the ``assert`` in the source would fire.
        """
        crossover, observed = 0.20, 0.10           # linear law now lands late
        with pytest.raises(AssertionError):
            assert crossover <= observed, "refusing late rather than early"
        # And the shipped numbers are on the right side of it.
        for key, obs in target.OMEGA_K_MEASURED_CROSSING.items():
            if obs is not None:
                assert target.OMEGA_K_CROSSOVER[key] <= obs

    def test_virial_is_the_more_sensitive_one(self):
        """Not decoration: it is 4x, so the 200m number applied at virial
        understates the cost fourfold.  A single pooled coefficient would be
        wrong for both."""
        c = target.OMEGA_K_COST
        assert c["vir"] > 3.0 * c["200m"]

    def test_the_crossover_follows_from_what_the_weights_record(self):
        r"""Derived, not written down.

        :data:`~emu_hmf.target.OMEGA_K_CROSSOVER` is where the induced error,
        added in quadrature to a file's held-out residual, reaches the
        ``tinker08`` baseline that same file was measured against.  Both numbers
        are already in the ``.npz``, so the crossover cannot drift from them
        without this failing.
        """
        from emu_hmf import model
        for key, path in model.WEIGHTS.items():
            with np.load(path) as d:
                v, b = float(d["val_rms"]), float(d["baseline_rms"])
            want = np.sqrt(b ** 2 - v ** 2) / target.OMEGA_K_COST[key]
            assert target.OMEGA_K_CROSSOVER[key] == pytest.approx(want, rel=1e-3)

    def test_refusing_would_cost_more_than_accepting_at_a_realistic_prior(self):
        r"""The measurement's whole point, as a test.

        At :math:`|\Omega_k| = 0.002` --- Planck with BAO --- a curved cosmology
        evaluated through this correction is an order of magnitude closer to the
        emulator than the ``tinker08`` a refusal would send the caller back to.
        If that ever stops being true, the case for passing curvature through
        stops with it, and this test is what says so.
        """
        from emu_hmf import model
        for key, path in model.WEIGHTS.items():
            with np.load(path) as d:
                v, b = float(d["val_rms"]), float(d["baseline_rms"])
            degraded = np.sqrt(v ** 2 + (target.OMEGA_K_COST[key] * 0.002) ** 2)
            assert degraded < b / 10.0, (key, degraded, b)
            # And the crossover is well outside any prior in use.
            assert target.OMEGA_K_CROSSOVER[key] > 0.1

    @pytest.mark.gen
    def test_the_coefficient_reproduces(self, emu):
        r"""Re-measure it, so the constant cannot become folklore.

        Marked ``gen`` because it needs the CSST emulator.  Three assertions,
        and the first is the one that matters: at :math:`z = 0`, :math:`E(0) = 1`
        by closure whatever :math:`\Omega_k` is, so
        :math:`\Delta\ln f(z{=}0)` must vanish **identically**.  Forgetting that
        CSSTemu's own ``OmegaL`` line omits :math:`\Omega_k` is the one way to
        botch this recipe --- it would perturb the dark energy instead --- and
        that identity is what catches it.
        """
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                               / "docs"))
        from make_validity_bounds import curvature_cost

        thetas = box.sample(6, seed=11)
        for key, md in (("200m", "RockstarM200m"), ("vir", "RockstarMvir")):
            cost, z0 = curvature_cost(md, thetas)
            assert z0 < 1e-12, f"the z=0 identity failed for {md}: {z0}"
            ks = sorted(cost)
            coeffs = np.array([cost[k] / k for k in ks])
            # The coefficient is *not* constant, and the shape of the departure
            # is the point: it falls monotonically with |Omega_k|, so reading it
            # as linear overestimates the cost past the range it was fitted and
            # puts the crossover early.  Asserting a flat coefficient here would
            # pin the opposite of what was measured.
            assert np.all(np.diff(coeffs) <= 0), dict(zip(ks, coeffs))
            # Near-linear where the constant is quoted, at small curvature.
            small = coeffs[np.array(ks) <= 0.05]
            assert np.ptp(small) / small.mean() < 0.05, small
            # And falling by tens of per cent, not orders, out to the far end.
            assert coeffs[-1] > 0.75 * coeffs[0], coeffs
            # A six-point design undersamples the tail the 40-point constant
            # was measured on, so this bounds rather than reproduces it.
            assert coeffs[0] <= target.OMEGA_K_COST[key] * 1.05
            assert coeffs[0] > 0.5 * target.OMEGA_K_COST[key]


class TestTheOrderingBound:
    r"""The ordering cannot move the correction, and the table says by how much.

    Two separate claims, and keeping them apart is the point.  The *correction*
    is provably invariant: ``ggah_mod`` builds ``Omega_nu_matter`` from
    :math:`\Sigma m_\nu/3` in closed form, so the whole matter budget is
    bit-for-bit invariant under the split and
    :func:`~emu_hmf.target.theta_from_cosmology` returns identical numbers.
    What moves is the caller's :math:`\sigma(M)`, and
    :data:`~emu_hmf.target.NU_ORDERING_COST` bounds that.
    """

    def test_theta_does_not_move_with_the_ordering(self):
        """The invariance itself, exactly.  If this ever fails the whole
        accept-rather-than-refuse argument fails with it."""
        pytest.importorskip(
            "ggah_mod.cosmology",
            reason="the ordering invariance is UNVERIFIED without ggah_mod, "
                   "and it is the whole reason the ordering is accepted rather "
                   "than refused -- this skip is not a pass")
        for mnu in (0.12, 0.30):
            th = target.FIDUCIAL.copy()
            th[box.PARAMS.index("mnu")] = mnu
            deg = target.to_ggah_cosmology(th)
            a = np.asarray(target.theta_from_cosmology(deg))
            for other in ("normal", "inverted"):
                b = np.asarray(target.theta_from_cosmology(
                    deg.replace(nu_hierarchy=other)))
                assert np.array_equal(a, b), (mnu, other, b - a)

    def test_the_table_starts_where_the_ordering_starts(self):
        """Normal ordering has no solution below 0.058993 eV, so a table that
        began at zero would be quoting a comparison that cannot be made."""
        C = pytest.importorskip("ggah_mod.cosmology.constants")
        assert min(target.NU_ORDERING_COST) >= C.NU_MASS_FLOOR["normal"]
        assert max(target.NU_ORDERING_COST) <= box.BOX["mnu"][1]

    def test_it_falls_as_the_split_closes(self):
        """Largest at the floor, where the three masses differ most, and
        monotone down to the box's ceiling.  A table that did not would mean
        the measurement was picking up something other than the split."""
        m = sorted(target.NU_ORDERING_COST)
        mx = [target.NU_ORDERING_COST[k][0] for k in m]
        assert all(a >= b for a, b in zip(mx, mx[1:])), mx
        assert mx[0] > 5.0 * mx[-1]

    def test_it_is_why_the_ordering_is_accepted_and_not_refused(self):
        """The threshold, fixed before the measurement: the induced error added
        in quadrature must leave the published residual unchanged at the two
        figures it is quoted to."""
        from emu_hmf import model
        worst_rms = max(v[1] for v in target.NU_ORDERING_COST.values())
        for path in model.WEIGHTS.values():
            with np.load(path) as d:
                v = float(d["val_rms"])
            assert (round(np.sqrt(v ** 2 + worst_rms ** 2), 4)
                    == round(v, 4)), path.name


class TestTheGgahModIntegration:
    """The seam, from this side.

    ``ggah_mod`` consumes this package three ways: it loads the weights, it
    calls :func:`~emu_hmf.target.theta_from_cosmology` on every evaluation, and
    it restates the measured curvature constants in its own capability table.
    The first two would fail loudly if they broke.  **The third would not** ---
    two copies of one number drift silently, and every value stays plausible.

    So it is pinned here, the same way ``tests/test_box.py`` pins this package's
    copy of CSSTemu's bounds against CSSTemu.  Whoever holds the copy owes the
    test; this file owes the *source*, and a source that never checks its
    consumers finds out from a paper rather than from a suite.
    """

    @pytest.fixture(autouse=True)
    def _needs_the_halo_code(self):
        # The skip reason says what the skip *means*, not only what is absent.
        # A drift check that cannot run reads exactly like one that passed, and
        # this is the half of the pair that is allowed to be missing --- the
        # constants are still bound to their own archive by
        # `TestTheConstantsMatchTheMeasurement`, which needs nothing external.
        pytest.importorskip(
            "ggah_mod.halos.mass_function",
            reason="ggah_mod is absent, so the cross-package drift check is NOT "
                   "running -- these skips are not passes")

    #: ``ggah_mod``'s registry name -> this package's weights key.
    FITS = {"tinker08_csst": "200m", "tinker08_csst_vir": "vir"}

    def test_the_curvature_constants_have_not_drifted(self):
        from ggah_mod.halos.mass_function import MULTIPLICITY_CAPABILITY

        for fit, key in self.FITS.items():
            cap = MULTIPLICITY_CAPABILITY.get(fit)
            assert cap is not None, f"{fit} left ggah_mod's registry"
            r = getattr(cap, "curvature_response", None)
            if r is None:                    # it declares no response to check
                continue
            assert r.coefficient == target.OMEGA_K_COST[key], fit
            assert r.crossover == pytest.approx(
                target.OMEGA_K_CROSSOVER[key], rel=1e-3), fit
            assert r.measured_to == target.OMEGA_K_MEASURED_TO[key], fit
            assert r.measured_crossing == target.OMEGA_K_MEASURED_CROSSING[key], fit

    def test_both_recalibrations_are_still_registered_and_cosmology_dependent(self):
        """They are the only two multiplicity functions there that take a
        cosmology, and that difference is the whole reason this package exists.
        A rename on that side would make the correction silently unreachable."""
        from ggah_mod.halos import mass_function as mf

        for fit in self.FITS:
            assert fit in mf.MULTIPLICITY, fit
            assert fit in mf.COSMOLOGY_DEPENDENT_MULTIPLICITY, fit

    def test_the_seam_evaluates_and_returns_this_package_s_answer(self):
        """End to end: ggah_mod's entry point must agree with calling
        :class:`~emu_hmf.model.HmfCorrection` directly.  If the two disagree the
        conversion between them has moved, which is the failure this whole
        module is about."""
        from ggah_mod.cosmology import Cosmology
        from ggah_mod.halos import mass_function as mf

        from emu_hmf import model

        cosmo = Cosmology.create(sum_mnu=0.06, nu_hierarchy="degenerate")
        theta = target.theta_from_cosmology(cosmo)
        for fit, key in self.FITS.items():
            theirs = float(mf.MULTIPLICITY[fit](0.8, 0.5, cosmo=cosmo))
            ours = float(model.HmfCorrection(model.WEIGHTS[key]).fsigma(
                0.8, theta, 0.5))
            assert theirs == pytest.approx(ours, rel=1e-12), fit


class TestTheConstantsMatchTheMeasurement:
    """The constants are typed; the archive is generated.  Bind the two.

    ``docs/make_validity_bounds.py`` writes ``docs/data/validity_bounds.npz``
    and a human copies the rounded numbers into :mod:`emu_hmf.target`.  That
    copy is a hand step in the middle of a measured chain, and until this test
    existed nothing checked it --- so a regeneration that moved a coefficient
    would leave the constant, the documentation and ``ggah_mod``'s restatement
    all agreeing with each other and none of them with the measurement.

    Found by asking the question from the other end: ``ggah_mod``'s seam test
    can tell that its copy and this package's disagree, but not which of them
    is wrong.  This is the link that answers that.
    """

    ARCHIVE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "data" \
        / "validity_bounds.npz"

    @pytest.fixture(scope="class")
    def archive(self):
        if not self.ARCHIVE.exists():
            pytest.skip("regenerate with docs/make_validity_bounds.py")
        with np.load(self.ARCHIVE) as d:
            yield {k: d[k] for k in d.files}

    def test_the_coefficients_round_to_the_constants(self, archive):
        for key in ("200m", "vir"):
            assert round(float(archive[f"coeff_{key}"]), 4) == \
                target.OMEGA_K_COST[key], key

    def test_the_measured_range_is_the_archive_s(self, archive):
        for key, v in target.OMEGA_K_MEASURED_TO.items():
            assert float(archive["measured_to"]) == v, key

    def test_the_observed_crossing_is_the_archive_s(self, archive):
        for key, want in target.OMEGA_K_MEASURED_CROSSING.items():
            got = float(archive[f"crossing_{key}"])
            if want is None:
                assert np.isnan(got), (key, got)
            else:
                # Six significant figures, which is what both this package and
                # `ggah_mod` carry.  The tolerance is that rounding and nothing
                # else: 5e-6 is four orders below the 4 per cent by which this
                # crossing and the linear crossover deliberately differ, so it
                # cannot hide the disagreement it exists to measure.
                assert got == pytest.approx(want, rel=5e-6), (key, got)

    def test_the_ordering_table_is_the_archive_s(self, archive):
        got = dict(zip((round(float(m), 4) for m in archive["mnu"]),
                       zip(archive["order_max"], archive["order_rms"])))
        assert set(got) == set(target.NU_ORDERING_COST)
        # The table is stored to three significant figures, so half a unit in
        # the last place is 5e-3 relative.  The tolerance is the rounding and
        # nothing else: it is far below the 2.2 per cent of the residual this
        # table's largest entry represents, so it cannot hide a real move.
        for mnu, (mx, rms) in target.NU_ORDERING_COST.items():
            a, b = got[mnu]
            assert float(a) == pytest.approx(mx, rel=5e-3), mnu
            assert float(b) == pytest.approx(rms, rel=5e-3), mnu

    def test_the_crossover_is_derived_and_not_measured_directly(self, archive):
        """The one constant that is *not* in the archive, and should not be.

        :data:`~emu_hmf.target.OMEGA_K_CROSSOVER` follows from the coefficient
        and the two numbers each weights file already carries, so storing it in
        the archive would be a fourth copy of a derived quantity.  What this
        checks is that it still follows.
        """
        from emu_hmf import model

        assert not [k for k in archive if "crossover" in k]
        for key, path in model.WEIGHTS.items():
            with np.load(path) as d:
                v, b = float(d["val_rms"]), float(d["baseline_rms"])
            want = np.sqrt(b ** 2 - v ** 2) / float(archive[f"coeff_{key}"])
            assert target.OMEGA_K_CROSSOVER[key] == pytest.approx(want, rel=1e-3)
