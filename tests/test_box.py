r"""The box is CSST's, and the copy must not drift from it."""
import numpy as np
import pytest

from emu_hmf import box


class TestTheBoxIsTheEmulatorsOwn:
    """Copied rather than imported, and therefore checked.

    CSSTemu is a ``[gen]`` dependency: a package meant to be importable
    by a forecast cannot make the forecast install a Gaussian-process emulator
    to find out what its own bounds are.  So the numbers are written down here
    -- and this test is the reason that is safe.
    """

    def test_it_matches_csstemu_exactly(self):
        pytest.importorskip("ggah_mod.halos._cemulator_compat",
                            reason="it is what makes CSSTemu importable")
        CE = pytest.importorskip("CEmulator.Emulator",
                                 reason="CSSTemu is a [gen] dependency")
        theirs = CE.HMF_CEmulator().param_limits
        assert set(theirs) == set(box.BOX), (
            f"CSSTemu's parameters are {sorted(theirs)}, ours {sorted(box.BOX)}")
        for p, (lo, hi) in box.BOX.items():
            assert (lo, hi) == tuple(theirs[p]), (
                f"{p}: ours {(lo, hi)}, CSSTemu's {tuple(theirs[p])}")

    def test_the_column_order_is_theirs_too(self):
        pytest.importorskip("ggah_mod.halos._cemulator_compat",
                            reason="it is what makes CSSTemu importable")
        CE = pytest.importorskip("CEmulator.Emulator",
                                 reason="CSSTemu is a [gen] dependency")
        assert list(box.PARAMS) == list(CE.HMF_CEmulator().param_names)


class TestTheDesign:
    def test_every_point_is_inside(self):
        d = box.sample(64)
        lo = np.array([box.BOX[p][0] for p in box.PARAMS])
        hi = np.array([box.BOX[p][1] for p in box.PARAMS])
        assert np.all(d >= lo) and np.all(d <= hi)

    def test_it_is_a_latin_hypercube(self):
        """One point per stratum in every column, which is the whole property."""
        n = 40
        d = box.sample(n)
        lo = np.array([box.BOX[p][0] for p in box.PARAMS])
        hi = np.array([box.BOX[p][1] for p in box.PARAMS])
        u = (d - lo) / (hi - lo)
        for j in range(u.shape[1]):
            strata = np.floor(u[:, j] * n).astype(int)
            assert len(set(strata)) == n, f"column {box.PARAMS[j]} is not stratified"

    def test_it_is_reproducible_from_the_seed(self):
        np.testing.assert_array_equal(box.sample(16), box.sample(16))
        assert not np.array_equal(box.sample(16), box.sample(16, seed=1))

    def test_the_dark_energy_corner_needs_no_rejection(self):
        """`emu_pk` rejects `w0 + wa >= 0`; here it cannot arise.

        Stated as a test rather than a comment because a guard that never fires
        and a guard that is missing look identical in the source.
        """
        assert box.BOX["w"][1] + box.BOX["wa"][1] < 0.0
        d = box.sample(256)
        w = d[:, box.PARAMS.index("w")]
        wa = d[:, box.PARAMS.index("wa")]
        assert np.all(w + wa < 0.0)


class TestTheRefusal:
    def test_it_names_every_offender_not_the_first(self):
        with pytest.raises(ValueError) as e:
            box.check({"H0": 55.0, "mnu": 0.4, "ns": 0.95})
        msg = str(e.value)
        assert "H0" in msg and "mnu" in msg and "ns" not in msg

    def test_the_bounds_are_closed(self):
        for p, (lo, hi) in box.BOX.items():
            box.check({p: lo})
            box.check({p: hi})

    #: The two boxes are stated in different variables, so comparing them means
    #: mapping this one into `emu_pk`'s.  One place, used by every test below.
    @staticmethod
    def _as_emu_pk(design):
        Ob, Ocb, H0, ns, A, w, wa, mnu = np.asarray(design).T
        h = H0 / 100.0
        return {"omega_b": Ob * h ** 2, "omega_cdm": (Ocb - Ob) * h ** 2,
                "h": h, "n_s": ns, "ln10A_s": np.log(10.0 * A),
                "sum_mnu": mnu, "w0": w, "wa": wa}

    def test_it_is_narrower_than_emu_pk_on_seven_of_the_eight_shared_axes(self):
        """The sigma(M) this recalibrates against comes from `emu_pk`, so the
        two boxes have to be compared rather than assumed compatible."""
        emu_pk_box = pytest.importorskip("emu_pk.box",
                                         reason="emu_pk is the sigma(M) source")
        mapped = self._as_emu_pk(box.sample(2000))
        inside = [p for p, v in mapped.items()
                  if v.min() >= emu_pk_box.BOX[p][0]
                  and v.max() <= emu_pk_box.BOX[p][1]]
        assert set(mapped) - set(inside) == {"omega_b"}, sorted(inside)

    def test_the_one_axis_it_escapes_on_is_omega_b(self):
        """And it escapes on *both* sides, which is the whole reason the
        training set is generated with CLASS and not with a network spectrum.

        `generate.py` quotes 30 per cent of the box as uncovered; that number
        lives here so it cannot drift from the boxes it is a property of.
        """
        emu_pk_box = pytest.importorskip("emu_pk.box",
                                         reason="emu_pk is the sigma(M) source")
        ob = self._as_emu_pk(box.sample(4000))["omega_b"]
        lo, hi = emu_pk_box.BOX["omega_b"]
        assert ob.min() < lo and ob.max() > hi, (ob.min(), ob.max())
        covered = ((ob >= lo) & (ob <= hi)).mean()
        assert 0.65 < covered < 0.75, covered

    def test_emu_pk_carries_axes_this_box_does_not_have_at_all(self):
        """Not a narrowing but an absence.

        `emu_pk` 2.0.0 samples curvature and a neutrino mass split; CSST's suite
        is flat with one species, so there is nothing here to bound.  Asserted
        because the distinction decides what this package can do about them:
        a narrower axis could be widened by retraining, an absent one cannot.
        """
        emu_pk_box = pytest.importorskip("emu_pk.box",
                                         reason="emu_pk is the sigma(M) source")
        # The three axes arrived in emu_pk 2.0.0.  Skipping rather than passing
        # on 1.x is the point: a version-blind assertion here would go green
        # against a box that has none of them and report agreement with a
        # release this statement is not about.
        pytest.importorskip("emu_pk", minversion="2.0.0",
                            reason="the three extra axes arrived in emu_pk 2.0")
        extra = set(emu_pk_box.PARAMS) - set(self._as_emu_pk(box.sample(4)))
        assert extra == {"Omega_k", "nu_r1", "nu_r2"}, extra
        assert not extra & set(box.PARAMS)
