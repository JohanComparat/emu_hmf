r"""Assembling a training set, and fitting a correction to it.

Nothing here runs CLASS or the CSST emulator: the shards are synthesised, so
the pipeline that turns them into weights is testable in the same environment
the released package installs into.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from emu_hmf import box, fit, model, target


def _write_shard(path, theta, z, sigma, f, massdef=target.DEFAULT_MASSDEF,
                 failed=()):
    """One shard in the schema :func:`emu_hmf.generate.shard` writes."""
    np.savez(path,
             idx=np.arange(len(theta), dtype=np.int64),
             theta=np.asarray(theta, dtype=np.float64),
             f=np.asarray(f, dtype=np.float32),
             sigma=np.asarray(sigma, dtype=np.float32),
             dlns=np.full(np.shape(f), -0.3, dtype=np.float32),
             z=np.asarray(z, dtype=np.float64),
             m=np.logspace(12, 14, np.shape(f)[-1]),
             failed_idx=np.array(sorted(failed), dtype=np.int64),
             massdef=np.array(str(massdef)))


def _synthetic_campaign(directory, n_shards=2, n_cosmo=12, n_m=12,
                        g_true=None, seed=0):
    r"""A training set generated from a *known* correction.

    ``f`` is Tinker08 with :math:`g` set by the cosmology, so a fit that works
    has something specific to recover and the test is not merely asserting that
    a loss went down.
    """
    z = np.array(target.Z_TRAINED)
    nu = np.linspace(*target.NU_TRUSTED, n_m)
    sigma = target.DELTA_C / nu
    paths = []
    for i in range(n_shards):
        theta = np.array([box.sample(1, seed=1000 * i + s + seed)[0]
                          for s in range(n_cosmo)])
        f = np.empty((n_cosmo, len(z), n_m))
        for ic, th in enumerate(theta):
            g = g_true(th)
            for iz, zv in enumerate(z):
                f[ic, iz] = np.asarray(model.tinker08(sigma, zv, g))
        path = f"{directory}/hmf_{i:03d}.npz"
        _write_shard(path, theta,
                     z, np.broadcast_to(sigma, f.shape).copy(), f)
        paths.append(path)
    return paths


class TestLoadShards:
    def test_it_refuses_a_directory_with_no_shards(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="hmf_"):
            fit.load_shards(str(tmp_path))

    def test_it_keeps_only_the_trusted_peak_heights(self, tmp_path,
                                                    make_shards):
        """The cut is a drop, not a weight of zero: the reason for it is that
        the target is not trustworthy there, which is not a statement about
        weighting."""
        lo, hi = target.NU_TRUSTED
        inside = target.DELTA_C / (0.5 * (lo + hi))
        outside = target.DELTA_C / (hi + 5.0)
        sigma = np.array([inside, outside, inside, outside, inside])
        make_shards(tmp_path, n_cosmo=2, n_z=3, n_m=5, sigma=sigma)
        d = fit.load_shards(str(tmp_path))
        assert len(d.ln_f) == 2 * 3 * 3, "the out-of-range masses survived"
        nu = target.DELTA_C / d.sigma
        assert np.all((nu >= lo) & (nu <= hi))

    def test_it_drops_rows_that_are_not_finite_and_positive(self, tmp_path,
                                                            make_shards):
        f = np.array([0.2, np.nan, 0.0, -1.0, np.inf])
        make_shards(tmp_path, n_cosmo=2, n_z=3, n_m=5, f=f)
        d = fit.load_shards(str(tmp_path))
        assert len(d.ln_f) == 2 * 3 * 1
        assert np.all(np.isfinite(d.ln_f))

    def test_it_records_which_design_each_row_came_from(self, tmp_path,
                                                        make_shards):
        """A held-out row is not a held-out cosmology.

        Each design contributes a few hundred rows -- the trained redshifts
        times the masses inside the peak-height cut -- and at fixed cosmology
        ``ln f`` is a smooth function of ``sigma``.  Split at random over rows
        and the network validates by interpolating between neighbouring masses
        of a design it trained on, which is a real measurement of something and
        not the one anybody wants.  This correction is only ever asked for a
        cosmology it has not seen.
        """
        make_shards(tmp_path, n_cosmo=3, n_z=4, n_m=5)
        d = fit.load_shards(str(tmp_path))
        assert set(np.unique(d.cosmo_id).tolist()) == {0, 1, 2}
        assert np.bincount(d.cosmo_id).tolist() == [4 * 5] * 3

    def test_the_ids_stay_unique_across_shards(self, tmp_path, make_shards):
        """Two shards each numbering their designs from zero would collide,
        and the split would then hold out design 7 of shard 0 while training on
        design 7 of shard 1 -- a different cosmology with the same label, which
        looks like a clean split and is not one."""
        make_shards(tmp_path, n_shards=2, n_cosmo=2, n_z=3, n_m=4)
        d = fit.load_shards(str(tmp_path))
        assert set(np.unique(d.cosmo_id).tolist()) == {0, 1, 2, 3}

    def test_it_carries_its_own_provenance(self, tmp_path, make_shards):
        make_shards(tmp_path, n_shards=2, n_cosmo=3, n_z=4, n_m=5,
                    failed=(9,))
        d = fit.load_shards(str(tmp_path))
        assert d.provenance["massdef"] == target.DEFAULT_MASSDEF
        assert d.provenance["n_shards"] == 2
        assert d.provenance["n_cosmologies"] == 6
        assert d.provenance["n_refused"] == 2          # one per shard
        assert d.provenance["n_rows"] == len(d.ln_f)
        assert (d.provenance["nu_lo"], d.provenance["nu_hi"]) == \
            target.NU_TRUSTED

    def test_the_arrays_are_all_the_same_length(self, tmp_path, make_shards):
        make_shards(tmp_path, n_shards=2, n_cosmo=3, n_z=4, n_m=5)
        d = fit.load_shards(str(tmp_path))
        n = len(d.ln_f)
        assert len(d.z) == len(d.sigma) == len(d.cosmo_id) == n
        assert d.theta.shape == (n, len(box.PARAMS))

    def test_a_narrower_range_can_be_asked_for(self, tmp_path, make_shards):
        sigma = target.DELTA_C / np.array([0.6, 1.5, 2.9])
        make_shards(tmp_path, n_cosmo=1, n_z=1, n_m=3, sigma=sigma)
        assert len(fit.load_shards(str(tmp_path), nu_range=(1.0, 2.0)).ln_f) == 1


class TestOneDefinitionPerFit:
    """The correction is a correction *to Tinker08 at a particular Delta*.

    Averaging shards from two definitions would fit a correction for neither,
    and every number in the result would look entirely reasonable.
    """

    def test_mixed_definitions_are_refused(self, tmp_path):
        theta, z = box.sample(2), np.array([0.0, 1.0])
        sigma = np.full((2, 2, 3), target.DELTA_C / 1.5)
        f = np.full((2, 2, 3), 0.2)
        for i, md in enumerate(("RockstarM200m", "RockstarMvir")):
            _write_shard(tmp_path / f"hmf_{i:03d}.npz", theta, z, sigma, f,
                         massdef=md)
        with pytest.raises(ValueError, match="mixes halo definitions"):
            fit.load_shards(str(tmp_path))

    def test_a_shard_that_cannot_name_its_definition_is_refused(self, tmp_path,
                                                                make_shards):
        """Silence is not the default.  A shard with no ``massdef`` could be
        either definition, and guessing produces a correction that is wrong by
        the size of the difference between them."""
        make_shards(tmp_path, n_shards=1)
        with np.load(tmp_path / "hmf_000.npz") as d:
            arrays = {k: d[k] for k in d.files if k != "massdef"}
        np.savez(tmp_path / "hmf_000.npz", **arrays)
        with pytest.raises(ValueError, match="does not say which halo"):
            fit.load_shards(str(tmp_path))


class TestTheFitStartsAtTinker08:
    def test_the_output_layer_is_initialised_to_zero(self):
        """So epoch zero *is* the published fit, and every step after it is a
        measured improvement on one.  A fit that cannot get worse than its own
        starting point is a different kind of object from one that can."""
        import jax

        sizes = [9, 8, 8, 4]
        p = fit._init(jax.random.PRNGKey(0), sizes)
        assert np.all(np.asarray(p["W2"]) == 0.0)
        assert np.all(np.asarray(p["b2"]) == 0.0)
        assert np.any(np.asarray(p["W0"]) != 0.0), "the hidden layers are dead"

    def test_an_untrained_network_reproduces_the_carrier(self):
        import jax

        p = fit._init(jax.random.PRNGKey(0), [9, 8, 4])
        x = model.normalise(target.FIDUCIAL, 0.5)
        g = model._mlp(p, x)
        np.testing.assert_allclose(np.asarray(g), 0.0, atol=0)
        sigma = np.linspace(0.6, 2.0, 8)
        np.testing.assert_array_equal(
            np.asarray(model.tinker08(sigma, 0.5, g[0])),
            np.asarray(model.tinker08(sigma, 0.5)))


@pytest.mark.slow
class TestTheFitEndToEnd:
    """Recover a correction that was put there on purpose."""

    @staticmethod
    def _g_true(theta):
        """A smooth, real function of the cosmology -- amplitude only."""
        u = (theta[1] - box.BOX["Omegam"][0]) / (
            box.BOX["Omegam"][1] - box.BOX["Omegam"][0])
        return np.array([0.10 * u - 0.05, 0.0, 0.0, 0.0])

    def test_it_finds_a_known_correction(self, tmp_path):
        _synthetic_campaign(tmp_path, n_shards=2, n_cosmo=24, n_m=12,
                            g_true=self._g_true)
        out = tmp_path / "weights.npz"
        fit.fit(str(tmp_path), out, hidden=(16, 16), epochs=60, batch=512,
                val_frac=0.25)
        with np.load(out) as w:
            assert float(w["val_rms"]) < 0.25 * float(w["baseline_rms"]), (
                "the fit did not improve on Tinker08 by much, on data that is "
                "Tinker08 times a correction it could express exactly")
            assert float(w["val_rms"]) < 0.01

    def test_the_weights_file_carries_everything_the_model_reads_back(
            self, tmp_path):
        _synthetic_campaign(tmp_path, n_shards=1, n_cosmo=12, n_m=8,
                            g_true=self._g_true)
        out = tmp_path / "weights.npz"
        fit.fit(str(tmp_path), out, hidden=(8,), epochs=5, batch=512,
                val_frac=0.25)
        c = model.HmfCorrection(out)          # the real consumer, not a mock
        assert np.all(np.isfinite(np.asarray(c.g(target.FIDUCIAL, 0.5))))
        with np.load(out) as w:
            for k in ("n_layers", "params_order", "massdef", "val_rms",
                      "baseline_rms", "nu_range", "n_cosmologies", "n_rows",
                      "n_val_cosmologies", "split_by_cosmology"):
                assert k in w, k
            assert list(w["params_order"]) == list(box.PARAMS) + ["z"]
            assert int(w["split_by_cosmology"]) == 1

    def test_the_fit_is_float64_whatever_the_environment(self, tmp_path):
        """Pinned inside ``fit`` rather than left to a flag set elsewhere: a
        network fitted in single precision and one fitted in double are two
        different sets of weights, and the residual being quoted is half a per
        cent."""
        _synthetic_campaign(tmp_path, n_shards=1, n_cosmo=8, n_m=8,
                            g_true=self._g_true)
        out = tmp_path / "weights.npz"
        fit.fit(str(tmp_path), out, hidden=(8,), epochs=3, batch=512,
                val_frac=0.25)
        with np.load(out) as w:
            assert w["W0"].dtype == np.float64

    def test_the_held_out_split_is_by_cosmology(self, tmp_path, capsys):
        _synthetic_campaign(tmp_path, n_shards=1, n_cosmo=8, n_m=6,
                            g_true=self._g_true)
        fit.fit(str(tmp_path), tmp_path / "w.npz", hidden=(8,), epochs=2,
                batch=256, val_frac=0.25)
        assert "held out 2 of 8 cosmologies entirely" in capsys.readouterr().out


class TestTheCommandLine:
    def test_it_passes_the_shard_directory_and_shape_through(self, tmp_path,
                                                             monkeypatch):
        seen = {}
        monkeypatch.setattr(fit, "fit",
                            lambda *a, **k: seen.update(args=a, kw=k))
        fit.main(["--shards", str(tmp_path), "--out", "w.npz",
                  "--epochs", "3", "--hidden", "8", "8"])
        assert seen["args"][0] == str(tmp_path)
        assert seen["kw"]["hidden"] == (8, 8)
        assert seen["kw"]["epochs"] == 3
