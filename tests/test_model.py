r"""The inference path: Tinker08, the correction, and the gradient through it.

Everything in this module runs with numpy and JAX alone.  That is deliberate
and is the point of the module: it is the only part of the package a user of
the *released* distribution touches, so it must be testable in the environment
that distribution creates.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from emu_hmf import box, model, target

FIDUCIAL = target.FIDUCIAL

#: The trusted peak-height range, expressed in the sigma the fit sees.
SIGMA_TRUSTED = (target.DELTA_C / target.NU_TRUSTED[1],
                 target.DELTA_C / target.NU_TRUSTED[0])
SIGMA_GRID = np.linspace(*SIGMA_TRUSTED, 32)


def _tinker08_reference(sigma, z=0.0, delta=200.0):
    """Tinker et al. (2008) eq. 3 and Table 2, written out in plain numpy.

    An independent restatement, not a call into the package: the module under
    test claims to carry the published fit unchanged, and a test that reached
    for ``model.T08`` to check ``model.tinker08`` would be asserting that a
    dictionary equals itself.
    """
    sigma = np.asarray(sigma, dtype=float)
    zp1 = 1.0 + np.asarray(z, dtype=float)
    alpha = 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))
    A = 0.186 * zp1 ** -0.14
    a = 1.47 * zp1 ** -0.06
    b = 2.57 * zp1 ** -alpha
    c = 1.19
    return A * ((sigma / b) ** -a + 1.0) * np.exp(-c / sigma ** 2)


class TestTheCarrierIsTinker08Unchanged:
    """The published fit, reproduced, because the whole design rests on it."""

    @pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 2.0, 3.0])
    def test_it_matches_the_published_fit(self, z):
        got = np.asarray(model.tinker08(SIGMA_GRID, z))
        np.testing.assert_allclose(got, _tinker08_reference(SIGMA_GRID, z),
                                   rtol=1e-12)

    def test_the_delta_200_exponent_is_the_published_one(self):
        """``alpha`` is a function of Delta, not a constant, and 200 is where
        this package pins it."""
        assert model.T08_ALPHA_200 == pytest.approx(
            10.0 ** (-((0.75 / np.log10(200.0 / 75.0)) ** 1.2)), rel=1e-15)
        assert model.T08_ALPHA_200 == pytest.approx(0.01068, abs=1e-5)

    def test_it_agrees_with_the_halo_model_code(self):
        """The restatement is safe only because this comparison exists.

        ``model.py`` writes Table 2 down rather than importing it, so that the
        module has no dependency beyond numpy and JAX.  That is a duplication,
        and this is what stops the two copies drifting.
        """
        mf = pytest.importorskip("ggah_mod.halos.mass_function",
                                 reason="the halo-model code is not installed")
        for z in (0.0, 1.0, 2.5):
            np.testing.assert_allclose(
                np.asarray(model.tinker08(SIGMA_GRID, z)),
                np.asarray(mf.fsigma_tinker08(SIGMA_GRID, z, delta=200.0)),
                rtol=1e-10)

    def test_zero_correction_is_the_carrier_exactly(self):
        """Not approximately.  ``g = 0`` must be a *point* in the same
        parameterisation, which is what makes "the correction is zero" a
        statement one can test rather than a limit one hopes for."""
        g = jnp.zeros(4)
        for z in (0.0, 1.0, 3.0):
            np.testing.assert_array_equal(
                np.asarray(model.tinker08(SIGMA_GRID, z, g)),
                np.asarray(model.tinker08(SIGMA_GRID, z)))


class TestTheShippedWeights:
    """Two files, and what each has to carry to be usable at all."""

    @pytest.mark.parametrize("key", sorted(model.WEIGHTS))
    def test_the_file_exists_and_is_loadable(self, key):
        assert model.WEIGHTS[key].exists(), model.WEIGHTS[key]
        w = model.load_weights(model.WEIGHTS[key])
        assert {"W0", "b0", "n_layers", "val_rms", "baseline_rms"} <= set(w)

    @pytest.mark.parametrize("key", sorted(model.WEIGHTS))
    def test_the_layer_count_matches_the_arrays(self, key):
        w = model.load_weights(model.WEIGHTS[key])
        n = int(w["n_layers"])
        assert {f"W{i}" for i in range(n)} <= set(w)
        assert {f"b{i}" for i in range(n)} <= set(w)
        assert f"W{n}" not in w, "more weight matrices than n_layers claims"

    @pytest.mark.parametrize("key", sorted(model.WEIGHTS))
    def test_the_input_order_is_the_boxs(self, key):
        w = model.load_weights(model.WEIGHTS[key])
        assert list(w["params_order"]) == list(box.PARAMS) + ["z"]
        assert w["W0"].shape[0] == len(box.PARAMS) + 1

    @pytest.mark.parametrize("key,massdef",
                             [("200m", "RockstarM200m"), ("vir", "RockstarMvir")])
    def test_the_file_names_the_definition_the_key_promises(self, key, massdef):
        """A weights file that does not know its own mass definition is a
        correction for no definition at all."""
        w = model.load_weights(model.WEIGHTS[key])
        assert str(w["massdef"]) == massdef
        assert massdef in target.MASSDEFS

    @pytest.mark.parametrize("key", sorted(model.WEIGHTS))
    def test_it_carries_what_it_was_fitted_on(self, key):
        """Provenance in the file, not in a terminal that has scrolled away."""
        w = model.load_weights(model.WEIGHTS[key])
        for k in ("n_cosmologies", "n_rows", "n_refused", "n_shards",
                  "nu_lo", "nu_hi", "val_frac", "epochs",
                  "n_val_cosmologies", "split_by_cosmology"):
            assert k in w, k
        assert int(w["split_by_cosmology"]) == 1, (
            "held out rows rather than cosmologies; the reported accuracy is "
            "then an interpolation error and not a generalisation error")
        assert (float(w["nu_lo"]), float(w["nu_hi"])) == target.NU_TRUSTED

    @pytest.mark.parametrize("key", sorted(model.WEIGHTS))
    def test_the_fit_improved_on_its_own_starting_point(self, key):
        w = model.load_weights(model.WEIGHTS[key])
        assert float(w["val_rms"]) < float(w["baseline_rms"])
        assert float(w["val_rms"]) < 0.01, "worse than one per cent in ln f"

    @pytest.mark.parametrize("key", sorted(model.WEIGHTS))
    def test_every_array_is_float64(self, key):
        """Both files, one precision.

        A network fitted in single precision and one fitted in double are two
        different sets of weights, and the residual being quoted is half a per
        cent.  ``fit.fit`` pins the mode; this is what keeps a file that
        escaped that pin from being shipped.
        """
        w = model.load_weights(model.WEIGHTS[key])
        for k, v in w.items():
            if v.dtype.kind == "f":
                assert v.dtype == np.float64, f"{key}:{k} is {v.dtype}"

    def test_the_weights_cannot_be_mutated_through_the_cache(self):
        """``load_weights`` is cached, so its arrays are shared by every
        caller; one that could write to them would change the correction
        everybody else evaluates, and every number would stay plausible."""
        w = model.load_weights(model.WEIGHTS["200m"])
        with pytest.raises(TypeError):
            w["W0"] = np.zeros(1)
        with pytest.raises(ValueError):
            w["W0"][0, 0] = 0.0

    def test_a_missing_file_says_how_to_make_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="emu_hmf.fit"):
            model.load_weights(tmp_path / "absent.npz")


class TestTheCorrectionAtTwoDefinitions:
    """What each of the two shipped corrections actually does to Tinker08."""

    def _ratio(self, key, z=0.0):
        """The correction, over the peak heights actually fitted at this ``z``.

        Not over :data:`~emu_hmf.target.NU_TRUSTED`: above :math:`z \\simeq
        0.25` the low-:math:`\\nu` half of that band is not in the training set
        at all, so a ratio measured there would be reporting an extrapolation.
        """
        nu = np.linspace(*target.nu_covered(z), 32)
        sigma = target.DELTA_C / nu
        c = model.HmfCorrection(model.WEIGHTS[key])
        f = np.asarray(c.fsigma(sigma, FIDUCIAL, z))
        return f / np.asarray(model.tinker08(sigma, z))

    def test_at_200m_and_z_zero_it_is_a_few_per_cent(self):
        """The headline claim, pinned against the shipped weights: at the
        definition Tinker08 was calibrated in, and at the redshift the offset
        was first noticed at, the emulator asks for a few per cent."""
        r = self._ratio("200m", 0.0)
        assert np.all(np.isfinite(r))
        assert 0.96 < r.min() and r.max() < 1.04, (r.min(), r.max())
        # And not identically one, or there would be nothing to ship.
        assert np.max(np.abs(r - 1.0)) > 0.005

    @pytest.mark.parametrize("z", [0.0, 1.0, 2.0, 3.0])
    def test_at_200m_it_stays_bounded_everywhere_it_was_fitted(self, z):
        r = self._ratio("200m", z)
        assert np.all(np.isfinite(r))
        assert 0.90 < r.min() and r.max() < 1.25, (z, r.min(), r.max())

    def test_the_correction_grows_with_redshift(self):
        """Which is why it is a function of ``z`` and not a number.

        Measured on the shipped weights, the rms of :math:`\\ln` (ratio) over
        the covered band runs 2.4, 5.7, 9.5 and 11.7 per cent at
        :math:`z = 0, 1, 2, 3`: a factor of five between the ends.  So the
        low-redshift figure quoted on its own would understate the correction
        several-fold over most of the range it is defined on.
        """
        size = [float(np.sqrt(np.mean(np.log(self._ratio("200m", z)) ** 2)))
                for z in (0.0, 1.0, 2.0, 3.0)]
        assert size == sorted(size), size
        assert size[0] < 0.03 and size[-1] > 0.10, size
        assert size[-1] / size[0] > 4.0, size

    def test_at_vir_and_z_zero_it_is_not_a_per_cent_correction(self):
        """The virial weights carry the boundary change as well as the
        recalibration, so they are *not* a small correction.

        ``tinker08`` is the carrier at :math:`\\Delta_{\\rm m}=200` in both
        cases; reading "correction" as "small" at ``vir`` is the misreading
        this test exists to make impossible.
        """
        r = self._ratio("vir", 0.0)
        assert np.all(np.isfinite(r))
        assert np.median(r) < 0.95, np.median(r)
        assert r.min() > 0.5, r.min()

    def test_the_two_are_different_functions(self):
        """Measured, not assumed -- which is why there are two files and not
        one function with a Delta argument."""
        a, b = self._ratio("200m"), self._ratio("vir")
        rms = float(np.sqrt(np.mean((a / b - 1.0) ** 2)))
        residual = max(float(model.load_weights(model.WEIGHTS[k])["val_rms"])
                       for k in model.WEIGHTS)
        assert rms > residual, (
            f"the two corrections disagree by {rms:.4f} rms, no more than the "
            f"{residual:.4f} residual either achieves; a single correction "
            "would then be defensible and two files would not be")
        assert rms > 0.05


class TestTheInputNormalisation:
    def test_the_box_corners_map_to_the_unit_cube(self):
        lo = np.array([box.BOX[p][0] for p in box.PARAMS])
        hi = np.array([box.BOX[p][1] for p in box.PARAMS])
        np.testing.assert_allclose(
            np.asarray(model.normalise(lo, 0.0))[0, :-1], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(model.normalise(hi, 0.0))[0, :-1], 1.0, atol=1e-12)

    def test_redshift_is_the_last_column_and_scales_by_z_max(self):
        x = np.asarray(model.normalise(FIDUCIAL, 3.0, z_max=3.0))
        assert x.shape == (1, len(box.PARAMS) + 1)
        assert x[0, -1] == pytest.approx(1.0)
        assert np.asarray(model.normalise(FIDUCIAL, 1.5))[0, -1] == \
            pytest.approx(0.5)

    def test_a_vector_of_redshifts_gives_a_row_each(self):
        z = np.array([0.0, 1.0, 2.0])
        x = np.asarray(model.normalise(FIDUCIAL, z))
        assert x.shape == (3, len(box.PARAMS) + 1)
        np.testing.assert_allclose(x[:, -1], z / 3.0)


@pytest.fixture(scope="module")
def corr():
    return model.HmfCorrection()


class TestTheGradient:
    """The reason this package exists rather than a table of numbers."""

    def test_ln_f_responds_to_every_one_of_the_eight(self, corr):
        """A flat direction here is a parameter the forecast cannot constrain
        through the mass function, so absence of response is the failure mode
        worth testing for by name."""
        def ln_f(theta):
            return jnp.log(corr.fsigma(1.0, theta, 0.5))

        g = np.asarray(jax.grad(ln_f)(jnp.asarray(FIDUCIAL)))
        assert np.all(np.isfinite(g))
        flat = [p for p, v in zip(box.PARAMS, g) if v == 0.0]
        assert not flat, f"no response to {flat}"

    def test_the_gradient_matches_a_finite_difference(self, corr):
        def ln_f(theta):
            return float(jnp.log(corr.fsigma(1.0, theta, 0.5)))

        auto = np.asarray(jax.grad(
            lambda t: jnp.log(corr.fsigma(1.0, t, 0.5)))(jnp.asarray(FIDUCIAL)))
        for i, p in enumerate(box.PARAMS):
            step = 1e-5 * (box.BOX[p][1] - box.BOX[p][0])
            up, dn = FIDUCIAL.copy(), FIDUCIAL.copy()
            up[i] += step
            dn[i] -= step
            fd = (ln_f(up) - ln_f(dn)) / (2 * step)
            assert auto[i] == pytest.approx(fd, rel=2e-4, abs=1e-8), p

    def test_it_is_differentiable_in_redshift_too(self, corr):
        g = float(jax.grad(lambda z: jnp.log(corr.fsigma(1.0, FIDUCIAL, z)))(0.5))
        assert np.isfinite(g) and g != 0.0

    def test_dndlnM_is_differentiable_and_has_the_right_form(self, corr):
        m, sig, dlns, rho = 1e13, 0.9, -0.35, 8.5e10
        got = float(corr.dndlnM(m, sig, dlns, rho, FIDUCIAL, 0.0))
        f = float(corr.fsigma(sig, FIDUCIAL, 0.0))
        assert got == pytest.approx(f * (rho / m) * abs(dlns), rel=1e-12)
        g = jax.grad(lambda t: jnp.log(
            corr.dndlnM(m, sig, dlns, rho, t, 0.0)))(jnp.asarray(FIDUCIAL))
        assert np.all(np.isfinite(np.asarray(g)))

    def test_it_survives_jit_and_vmap(self):
        """Both are how a forecast will actually call this: compiled once, and
        evaluated over a chain of cosmologies."""
        corr = model.HmfCorrection(check_box=False)
        f = jax.jit(lambda t: corr.fsigma(1.0, t, 0.5))
        assert np.isfinite(float(f(jnp.asarray(FIDUCIAL))))
        thetas = jnp.asarray(box.sample(8))
        out = np.asarray(jax.vmap(f)(thetas))
        assert out.shape == (8,) and np.all(np.isfinite(out))


class TestTheRefusal:
    def test_it_refuses_a_cosmology_outside_the_box(self):
        bad = FIDUCIAL.copy()
        bad[box.PARAMS.index("H0")] = 55.0
        with pytest.raises(ValueError, match="outside the CSST"):
            model.HmfCorrection().fsigma(1.0, bad, 0.0)

    def test_the_check_is_skipped_under_tracing(self):
        """Not a loophole: the values are not available inside a ``jit``, and
        raising there would break the gradient the package exists to provide.
        A jitted forward model is checked once, when it is built."""
        bad = FIDUCIAL.copy()
        bad[box.PARAMS.index("H0")] = 55.0
        f = jax.jit(lambda t: model.HmfCorrection().fsigma(1.0, t, 0.0))
        assert np.isfinite(float(f(jnp.asarray(bad))))

    def test_it_can_be_turned_off_deliberately(self):
        bad = FIDUCIAL.copy()
        bad[box.PARAMS.index("H0")] = 55.0
        v = model.HmfCorrection(check_box=False).fsigma(1.0, bad, 0.0)
        assert np.isfinite(float(v))
