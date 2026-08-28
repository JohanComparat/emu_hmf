r"""The target, and the two conversions that are silently wrong if guessed.

The module-level import guard is deliberately absent: only the classes that
genuinely need the CSST emulator or the halo-model code skip without them, so
that everything testable in a plain install keeps being tested.
"""
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
        # total sits above it by exactly Omega_nu.
        c = target.to_ggah_cosmology(th)
        assert float(c.Omega_cb) == pytest.approx(0.31, rel=1e-12)
        assert float(c.Omega_m) - float(c.Omega_cb) == pytest.approx(
            float(c.Omega_nu), rel=1e-12)
        assert float(c.Omega_cdm) == pytest.approx(0.31 - 0.049, rel=1e-12)
        if mnu > 0:
            assert float(c.Omega_m) > 0.31

    def test_a_box_edge_survives_the_round_trip(self, emu):
        """The corner where the convention matters most.

        :math:`\\Omega_m = 0.24` with :math:`\\Sigma m_\\nu = 0.3` is inside the
        box by construction; read the density convention the other way and the
        emulator refuses it as ``Omegam = 0.2329 < 0.24``.  A design point
        falling out of the box it was sampled inside is the only symptom this
        error has.
        """
        th = np.array([0.049, 0.24, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.30])
        target.set_cosmology(emu, th)
        assert float(emu.Cosmo.Omegam) == pytest.approx(0.24, abs=1e-12)

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

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30])
    def test_theta_to_cosmology_and_back(self, mnu):
        pytest.importorskip("ggah_mod.cosmology")
        th = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, mnu])
        back = np.asarray(target.theta_from_cosmology(
            target.to_ggah_cosmology(th)))
        assert back == pytest.approx(th, rel=1e-12), dict(
            zip(box.PARAMS, back - th))

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
