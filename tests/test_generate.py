r"""The shard writer, exercised without CLASS and without the CSST emulator.

Generation is the expensive half of this package -- some thirty-odd CPU-hours
for the shipped training set -- and it is the half whose failures are silent:
a shard that records the wrong mass definition, or loses a chunk to a kill, or
drops a refused cosmology without saying so, still loads and still fits.  So
the solver is stubbed and everything around it is tested.
"""
import sys
import types

import numpy as np
import pytest

from emu_hmf import box, generate, target


class _StubEmulator:
    """Stands in for ``HMF_CEmulator``: a mass function with the right shape.

    Deliberately not a mock that records calls.  What the tests below care
    about is the shard on disk, and a stub that returns a plausible
    ``dn/dlnM`` lets the real :func:`~emu_hmf.generate.solve_one` arithmetic
    run.
    """

    def __init__(self, raise_on=()):
        self.raise_on = set(raise_on)
        self.calls = []
        self.cosmos = []

    def set_cosmos(self, **kw):
        self.cosmos.append(kw)
        return self

    def get_dndlnM(self, z, M, massdef):
        self.calls.append((tuple(np.atleast_1d(z)), massdef))
        if massdef in self.raise_on:
            raise RuntimeError(f"emulator refused {massdef}")
        z = np.atleast_1d(z)
        return 1e-3 * (M[None, :] / 1e13) ** -1.9 * np.ones((len(z), 1))


class _StubPk:
    """A power spectrum with the right shape and a sane slope."""

    def pk_cb(self, k, z, cosmo):
        z = np.atleast_1d(z)
        return np.asarray(2e4 * k[None, :] ** -1.5 * np.ones((len(z), 1)))


@pytest.fixture
def stub_solve(monkeypatch):
    """Replace the CLASS-and-emulator solve with arithmetic of a known shape.

    Returns the list of design indices the solver was asked for, so a test can
    assert which slice of the design a shard covered.
    """
    seen = []

    def fake(theta, emu, pk, z=None, m=None, massdef=target.DEFAULT_MASSDEF):
        seen.append(np.asarray(theta).copy())
        n_z = len(target.Z_TRAINED if z is None else np.atleast_1d(z))
        n_m = len(generate.M_GRID if m is None else np.atleast_1d(m))
        if getattr(fake, "raise_at", None) is not None and \
                len(seen) - 1 in fake.raise_at:
            raise RuntimeError("CLASS refused this cosmology")
        return (np.full((n_z, n_m), 0.2), np.full((n_z, n_m), 1.1),
                np.full((n_z, n_m), -0.3))

    fake.raise_at = None
    monkeypatch.setattr(generate, "solve_one", fake)
    monkeypatch.setattr(target, "_emulator", lambda *a, **k: _StubEmulator())
    _stub_make_pk(monkeypatch)
    fake.seen = seen
    return fake


def _stub_make_pk(monkeypatch):
    """Give ``shard`` a spectrum without a Boltzmann solver behind it.

    ``shard`` reaches the halo code by ``from ggah_mod.cosmology.power import
    make_pk`` at call time, so when that package is absent the module has to be
    stood up in ``sys.modules`` rather than merely patched -- which is the
    point: this test module must pass in the environment ``pip install
    emu_hmf`` creates, where none of the generation stack exists.
    """
    try:
        from ggah_mod.cosmology import power
    except ImportError:
        power = types.ModuleType("ggah_mod.cosmology.power")
        cosmology = types.ModuleType("ggah_mod.cosmology")
        pkg = types.ModuleType("ggah_mod")
        cosmology.power, pkg.cosmology = power, cosmology
        for name, mod in (("ggah_mod", pkg),
                          ("ggah_mod.cosmology", cosmology),
                          ("ggah_mod.cosmology.power", power)):
            monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(power, "make_pk", lambda *a, **k: _StubPk(),
                        raising=False)


class TestTheGrids:
    def test_the_masses_are_the_range_the_target_is_trusted_over(self):
        assert generate.M_GRID[0] == pytest.approx(target.M_TRUSTED[0])
        assert generate.M_GRID[-1] == pytest.approx(target.M_TRUSTED[1])
        assert np.all(np.diff(generate.M_GRID) > 0)

    def test_the_wavenumbers_reach_well_past_what_the_masses_need(self):
        """The top-hat window has no sharp edge, so a grid that merely covers
        the Lagrangian radii of the masses truncates the variance integral.

        The thresholds are the halo code's own: ``k_max * R_min >= 10`` and
        ``k_min * R_max <= 0.1``.  Restated rather than imported, because this
        module must be readable without that package installed.
        """
        rho_cold = 0.26 * 2.775e11          # a low-Omega_cb corner of the box
        r = (3.0 * generate.M_GRID / (4.0 * np.pi * rho_cold)) ** (1.0 / 3.0)
        assert generate.K_GRID.max() * r.min() > 10.0
        assert generate.K_GRID.min() * r.max() < 0.1


class TestSolveOne:
    def test_it_converts_abundance_into_a_multiplicity_function(self):
        r"""``f = (dn/dlnM) M / (rho_cb |dln sigma/dln M|)``.

        Doing the conversion at generation is the whole design: the fit then
        absorbs this package's variance convention once, rather than leaving
        every caller to match it.
        """
        pytest.importorskip("ggah_mod.halos.variance",
                            reason="the variance integral lives there")
        emu, pk = _StubEmulator(), _StubPk()
        z = np.array([0.0, 1.0])
        m = np.logspace(12.0, 14.0, 6)
        f, sig, dlns = generate.solve_one(target.FIDUCIAL, emu, pk, z=z, m=m)
        assert f.shape == sig.shape == dlns.shape == (len(z), len(m))
        assert np.all(np.isfinite(f)) and np.all(f > 0)
        assert np.all(dlns < 0), "sigma must fall with mass"
        rho = float(target.to_ggah_cosmology(target.FIDUCIAL).rho_cold)
        dndlnm = np.asarray(emu.get_dndlnM(z=z, M=m, massdef="RockstarM200m"))
        np.testing.assert_allclose(
            f, dndlnm * m[None, :] / (rho * np.abs(dlns)), rtol=1e-10)

    def test_it_asks_the_emulator_for_the_definition_it_was_given(self):
        pytest.importorskip("ggah_mod.halos.variance")
        emu = _StubEmulator()
        generate.solve_one(target.FIDUCIAL, emu, _StubPk(),
                           z=np.array([0.0]), m=np.logspace(12, 13, 3),
                           massdef="RockstarMvir")
        assert emu.calls[-1][1] == "RockstarMvir"


class TestTheShardWriter:
    def test_it_writes_the_schema_the_fit_reads(self, tmp_path, stub_solve):
        out = tmp_path / "hmf_000.npz"
        generate.shard(0, 4, out, n_total=8)
        with np.load(out) as d:
            assert set(d.files) == {"idx", "theta", "f", "sigma", "dlns", "z",
                                    "m", "failed_idx", "massdef"}
            assert d["theta"].shape == (4, len(box.PARAMS))
            assert d["f"].shape == (4, len(target.Z_TRAINED),
                                    len(generate.M_GRID))
            assert d["idx"].tolist() == [0, 1, 2, 3]

    def test_it_stamps_the_mass_definition(self, tmp_path, stub_solve):
        """A shard that cannot name its own definition cannot be fitted: the
        correction is a correction to Tinker08 at a particular Delta, and a
        fit that averaged two would produce a correction for neither."""
        out = tmp_path / "hmf_000.npz"
        generate.shard(0, 2, out, n_total=4, massdef="RockstarMvir")
        with np.load(out) as d:
            assert str(d["massdef"]) == "RockstarMvir"

    def test_it_solves_the_slice_of_the_design_it_was_asked_for(self, tmp_path,
                                                                stub_solve):
        """The design is regenerated from the seed rather than read from a
        file, so two shards can never disagree about which index means which
        cosmology."""
        generate.shard(2, 3, tmp_path / "hmf_002.npz", n_total=12)
        design = box.sample(12)
        np.testing.assert_array_equal(np.array(stub_solve.seen),
                                      design[6:9])

    def test_the_last_shard_stops_at_the_end_of_the_design(self, tmp_path,
                                                           stub_solve):
        generate.shard(3, 4, tmp_path / "hmf_003.npz", n_total=14)
        with np.load(tmp_path / "hmf_003.npz") as d:
            assert d["idx"].tolist() == [12, 13]

    def test_a_shard_past_the_end_writes_nothing(self, tmp_path, stub_solve):
        generate.shard(9, 4, tmp_path / "hmf_009.npz", n_total=8)
        assert not (tmp_path / "hmf_009.npz").exists()
        assert stub_solve.seen == []

    def test_an_existing_shard_is_not_recomputed(self, tmp_path, stub_solve,
                                                 capsys):
        """A shard already on disk is left alone and its solves are not
        repeated, so an interrupted campaign resumes at the last chunk."""
        out = tmp_path / "hmf_000.npz"
        out.write_bytes(b"not really an npz")
        generate.shard(0, 4, out, n_total=8)
        assert stub_solve.seen == [], "it resolved a shard already on disk"
        assert "skipping" in capsys.readouterr().out
        assert out.read_bytes() == b"not really an npz"

    def test_it_writes_through_a_temporary_and_renames(self, tmp_path,
                                                       stub_solve):
        """So a kill during the write leaves the previous shard intact rather
        than a half-written file that loads and is wrong."""
        out = tmp_path / "hmf_000.npz"
        generate.shard(0, 2, out, n_total=4)
        assert out.exists()
        assert not list(tmp_path.glob("*.part.npz")), "the temporary survived"

    def test_it_writes_every_chunk_not_only_the_end(self, tmp_path,
                                                    stub_solve):
        out = tmp_path / "hmf_000.npz"
        generate.shard(0, 6, out, n_total=6, chunk=2)
        with np.load(out) as d:
            assert len(d["idx"]) == 6


class TestARefusalIsData:
    def test_a_failed_cosmology_is_recorded_not_dropped(self, tmp_path,
                                                        stub_solve):
        """Which design refused matters: it is regenerable from the seed, so a
        recorded index can be revisited, and a silently missing one cannot."""
        stub_solve.raise_at = {1}
        generate.shard(0, 4, tmp_path / "hmf_000.npz", n_total=8)
        with np.load(tmp_path / "hmf_000.npz") as d:
            assert d["failed_idx"].tolist() == [1]
            assert d["idx"].tolist() == [0, 2, 3]
            assert len(d["theta"]) == 3

    def test_the_refusal_is_reported_on_the_terminal_too(self, tmp_path,
                                                         stub_solve, capsys):
        stub_solve.raise_at = {0}
        generate.shard(0, 2, tmp_path / "hmf_000.npz", n_total=4)
        assert "RuntimeError" in capsys.readouterr().out


class TestTheCommandLine:
    def test_it_passes_the_shard_geometry_through(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(generate, "shard",
                            lambda *a, **k: seen.update(args=a, kw=k))
        generate.main(["--shard", "3", "--n-per-shard", "50", "--n-total",
                       "400", "--out", str(tmp_path / "x.npz"),
                       "--massdef", "RockstarMvir"])
        assert seen["args"][:2] == (3, 50)
        assert seen["args"][3] == 400
        assert seen["kw"]["massdef"] == "RockstarMvir"

    def test_it_refuses_a_definition_the_emulator_does_not_have(self, tmp_path):
        with pytest.raises(SystemExit):
            generate.main(["--shard", "0", "--out", str(tmp_path / "x.npz"),
                           "--massdef", "SomethingElse"])
