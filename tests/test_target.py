r"""The target, and the two conversions that are silently wrong if guessed."""
import numpy as np
import pytest

from emu_hmf import box, target

CE = pytest.importorskip("CEmulator.Emulator",
                         reason="CEmulator is a [gen] dependency")


@pytest.fixture(scope="module")
def emu():
    return target._emulator()


class TestTheDialectConversions:
    """`param_limits` and `set_cosmos` do not speak the same language.

    The bounds are stated in Omega_m and 1e9 A_s; the setter wants the *cold*
    density and A_s itself, which it then multiplies by 1e9 to check against
    those same bounds.  Guessing either is a plausible number rather than an
    error, which is why both are pinned here.
    """

    def test_the_amplitude_is_A_s_not_1e9_A_s(self, emu):
        """Passing the box's `A` straight through is caught by the emulator's
        own bound check -- so the test is that we do not do it."""
        with pytest.raises(ValueError, match="out of range"):
            emu.set_cosmos(Omegab=0.049, Omegac=0.261, H0=67.36,
                           As=2.1, ns=0.9649, w=-1.0, wa=0.0, mnu=0.06)

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30])
    def test_the_density_round_trip_is_exact(self, emu, mnu):
        """The quiet trap, pinned in both directions.

        CSST's `Omegam` -- what `param_limits` bounds -- is the *cold* density;
        `set_cosmos`'s `Omegac` argument is not, because the setter subtracts
        Omega_nu from it internally.  So the value handed over is the *total*
        matter minus baryons, and what comes back must be the sampled `Omegam`
        exactly.  Any other reading loses Omega_nu somewhere: half a per cent
        at 0.06 eV, two at 0.3, in the variance the whole recalibration is made
        against.
        """
        th = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, mnu])
        target.set_cosmology(emu, th)
        assert float(emu.Cosmo.Omegam) == pytest.approx(0.31, abs=1e-12)
        # And the identity that makes the two packages the same cosmology:
        # CSST's bounded `Omegam` is ggah_mod's *cold* density, so ggah_mod's
        # total sits above it by exactly Omega_nu.
        c = target.to_ggah_cosmology(th)
        assert float(c.Omega_cb) == pytest.approx(0.31, rel=1e-12)
        assert float(c.Omega_m) - float(c.Omega_cb) == pytest.approx(
            float(c.Omega_nu), rel=1e-12)
        assert float(c.Omega_cdm) == pytest.approx(0.31 - 0.049, rel=1e-12)
        if mnu > 0:
            assert float(c.Omega_m) > 0.31

    def test_a_box_edge_survives_the_round_trip(self, emu):
        """The corner that found the bug.

        `Omega_m = 0.24` with `Sigma m_nu = 0.3` is inside the box by
        construction; read the density convention the other way and CEmulator
        refuses it as `Omegam = 0.2329 < 0.24`.  A design point falling out of
        the box it was sampled inside is the only symptom this error has.
        """
        th = np.array([0.049, 0.24, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.30])
        target.set_cosmology(emu, th)
        assert float(emu.Cosmo.Omegam) == pytest.approx(0.24, abs=1e-12)

    def test_the_round_trip_reproduces_sigma8(self, emu):
        """The conversions, end to end, against the emulator's own sigma_8."""
        target.set_cosmology(emu, target._FIDUCIAL)
        s8 = float(np.atleast_1d(emu.get_sigma8())[0])
        assert 0.70 < s8 < 0.90, s8


class TestTheTargetIsARatio:
    """What has to be learned, and that it is small and smooth.

    The emulator ships *both* its simulation-calibrated dn/dlnM and its own
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
        r = self._ratio(emu, target._FIDUCIAL)
        assert np.all(np.isfinite(r))
        assert 0.90 < r.min() and r.max() < 1.10, r
        # And not identically one: if it were, there would be nothing to fit.
        assert np.max(np.abs(r - 1.0)) > 0.01, r

    def test_it_moves_with_the_cosmology(self, emu):
        """The whole premise of the recalibration, as a test.

        A correction that did not depend on cosmology would be a constant, and
        a constant is already inside Tinker08's amplitude.
        """
        base = self._ratio(emu, target._FIDUCIAL)
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
        d = np.asarray(target.csst_dndlnM(target._FIDUCIAL, 0.0, m,
                                          emu=emu)).ravel()
        t = np.asarray(target.csst_tinker08(target._FIDUCIAL, 0.0, m,
                                            emu=emu)).ravel()
        ln_r = np.log(d / t)
        coeff = np.polyfit(np.log(m), ln_r, 3)
        resid = np.max(np.abs(ln_r - np.polyval(coeff, np.log(m))))
        assert resid < 6e-3, resid

    def test_it_is_rough_above_the_trusted_range(self, emu):
        """And that is what `M_TRUSTED` is for.

        Stated as a test because a range that is never checked drifts into
        being a number someone chose.  The same cubic through the same ratio
        one decade higher is five times worse, and the reason is the target
        rather than the fit: a simulation suite runs out of clusters, and a
        Gaussian process is noisiest where its training data is thinnest.
        """
        m = np.logspace(12.0, 15.0, 25)
        d = np.asarray(target.csst_dndlnM(target._FIDUCIAL, 0.0, m,
                                          emu=emu)).ravel()
        t = np.asarray(target.csst_tinker08(target._FIDUCIAL, 0.0, m,
                                            emu=emu)).ravel()
        ln_r = np.log(d / t)
        coeff = np.polyfit(np.log(m), ln_r, 3)
        assert np.max(np.abs(ln_r - np.polyval(coeff, np.log(m)))) > 2e-2


class TestTheVarianceIsOurs:
    def test_sigma_falls_and_the_slope_is_negative(self):
        m = np.logspace(12.0, 15.0, 5)
        sig, dlns, rho = target.sigma_chain(target._FIDUCIAL, 0.0, m)
        assert np.all(np.diff(sig) < 0)
        assert np.all(dlns < 0)
        assert rho > 0

    def test_it_refuses_a_cosmology_outside_the_box(self, emu):
        bad = target._FIDUCIAL.copy()
        bad[box.PARAMS.index("H0")] = 55.0
        with pytest.raises(ValueError, match="outside the CSST"):
            target.csst_dndlnM(bad, 0.0, self_M := np.array([1e13]), emu=emu)


class TestTheMassDefinitionChoice:
    def test_the_default_is_the_spherical_overdensity_one(self):
        assert target.DEFAULT_MASSDEF == "RockstarM200m"
        assert "friends-of-friends" in target.MASSDEFS["FoFM200c"]

    def test_all_three_are_the_emulator_s(self, emu):
        for md in target.MASSDEFS:
            v = np.asarray(target.csst_dndlnM(
                target._FIDUCIAL, 0.0, np.array([1e13, 1e14]), massdef=md,
                emu=emu)).ravel()
            assert np.all(np.isfinite(v)) and np.all(v > 0), md

    def test_they_are_genuinely_different_masses(self, emu):
        """Not conventions: the wrapper in ggah_mod says so and this shows it."""
        m = np.array([1e13, 1e14])
        a, b = (np.asarray(target.csst_dndlnM(target._FIDUCIAL, 0.0, m,
                                              massdef=md, emu=emu)).ravel()
                for md in ("RockstarM200m", "FoFM200c"))
        assert np.max(np.abs(a / b - 1.0)) > 0.02, (a, b)


class TestTheValidationSplitIsHonest:
    """A held-out row is not a held-out cosmology.

    Each design contributes a few hundred rows -- twelve redshifts times the
    masses inside the peak-height cut -- and at fixed cosmology ``ln f`` is a
    smooth function of ``sigma``.  Split at random over rows and the network
    validates by interpolating between neighbouring masses of a design it
    trained on, which is a real measurement of something and not the one
    anybody wants.  This correction is only ever asked for a cosmology it has
    not seen.
    """

    def test_load_shards_reports_which_design_each_row_came_from(self, tmp_path):
        from emu_hmf import fit

        n_c, n_z, n_m = 3, 4, 5
        rng = np.random.default_rng(0)
        theta = np.array([box.sample(1, seed=s)[0] for s in (1, 2, 3)])
        np.savez(tmp_path / "hmf_000.npz",
                 theta=theta,
                 f=np.full((n_c, n_z, n_m), 0.2),
                 sigma=np.full((n_c, n_z, n_m), 1.686 / 1.5),
                 z=np.linspace(0.0, 1.0, n_z),
                 failed_idx=np.array([], dtype=np.int64))
        *_, cid = fit.load_shards(str(tmp_path))
        assert set(np.unique(cid).tolist()) == {0, 1, 2}
        # Every design contributes the same number of rows here, so a row split
        # would put two thirds of each design's rows in training.
        assert np.bincount(cid).tolist() == [n_z * n_m] * n_c

    def test_ids_stay_unique_across_shards(self, tmp_path):
        """The bug this offset exists for.

        Two shards each numbering their designs from zero would collide, and
        the split would then hold out design 7 of shard 0 while training on
        design 7 of shard 1 -- a different cosmology with the same label, which
        looks like a clean split and is not one.
        """
        from emu_hmf import fit

        n_z, n_m = 3, 4
        for i in (0, 1):
            theta = np.array([box.sample(1, seed=10 * i + s)[0] for s in (1, 2)])
            np.savez(tmp_path / f"hmf_{i:03d}.npz",
                     theta=theta,
                     f=np.full((2, n_z, n_m), 0.2),
                     sigma=np.full((2, n_z, n_m), 1.686 / 1.5),
                     z=np.linspace(0.0, 1.0, n_z),
                     failed_idx=np.array([], dtype=np.int64))
        *_, cid = fit.load_shards(str(tmp_path))
        assert set(np.unique(cid).tolist()) == {0, 1, 2, 3}, "shard ids collided"


class TestTheConversionRoundTrips:
    """Both directions, and the density that makes them differ.

    `to_ggah_cosmology` and `theta_from_cosmology` are the only two places the
    CSST/ggah_mod dialect is translated.  If they ever disagree, the halo layer
    evaluates the correction at a cosmology that is not the one it was asked
    for -- silently, because every value stays plausible.
    """

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30])
    def test_theta_to_cosmology_and_back(self, mnu):
        """The tolerance is the working precision's, not a fixed number.

        This asserted `rel=1e-12` unconditionally, which is a float64 claim
        made by a test that never said so: JAX defaults to float32, where the
        round trip is good to ~1e-7 and cannot be better.  It passed only for
        whoever happened to have `JAX_ENABLE_X64=1` in their environment.
        Deriving the tolerance from the dtype tests the conversion in the mode
        it is actually running in, in both modes.
        """
        import jax

        th = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, mnu])
        back = np.asarray(target.theta_from_cosmology(
            target.to_ggah_cosmology(th)))
        rel = 1e-12 if jax.config.jax_enable_x64 else 1e-6
        assert back == pytest.approx(th, rel=rel), dict(
            zip(box.PARAMS, back - th))

    def test_it_survives_tracing(self):
        """The reason it is not just `to_ggah_cosmology` inverted.

        This one is called from inside the halo layer's differentiable path,
        so every field may be a tracer.  A `float()` anywhere in it would raise
        a ConcretizationTypeError the first time somebody differentiated a mass
        function -- which is exactly the use it exists for.
        """
        import jax

        from ggah_mod.cosmology import PLANCK18

        def amp(a):
            return target.theta_from_cosmology(PLANCK18.replace(ln10A_s=a))[4]

        g = float(jax.grad(amp)(float(PLANCK18.ln10A_s)))
        # A = exp(ln10A_s)/10, so dA/dln10A_s = A.  Same precision caveat as
        # the round trip above: in float32 the gradient is exact to ~1e-6.
        rel = 1e-10 if jax.config.jax_enable_x64 else 1e-5
        assert g == pytest.approx(float(np.exp(PLANCK18.ln10A_s) / 10.0),
                                  rel=rel)

    def test_the_cold_density_is_what_crosses(self):
        """Not Omega_m: the box bounds the cold density."""
        from ggah_mod.cosmology import Cosmology

        c = Cosmology.create(sum_mnu=0.3)
        th = np.asarray(target.theta_from_cosmology(c))
        assert th[1] == pytest.approx(float(c.Omega_cb), rel=1e-12)
        assert th[1] < float(c.Omega_m)
