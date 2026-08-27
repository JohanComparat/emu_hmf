r"""The box is CSST's, and the copy must not drift from it."""
import numpy as np
import pytest

from emu_hmf import box


class TestTheBoxIsTheEmulatorsOwn:
    """Copied rather than imported, and therefore checked.

    ``CEmulator`` is a ``[gen]`` dependency: a package meant to be importable
    by a forecast cannot make the forecast install a Gaussian-process emulator
    to find out what its own bounds are.  So the numbers are written down here
    -- and this test is the reason that is safe.
    """

    def test_it_matches_CEmulator_exactly(self):
        CE = pytest.importorskip("CEmulator.Emulator",
                                 reason="CEmulator is a [gen] dependency")
        theirs = CE.HMF_CEmulator().param_limits
        assert set(theirs) == set(box.BOX), (
            f"CEmulator's parameters are {sorted(theirs)}, ours {sorted(box.BOX)}")
        for p, (lo, hi) in box.BOX.items():
            assert (lo, hi) == tuple(theirs[p]), (
                f"{p}: ours {(lo, hi)}, CEmulator's {tuple(theirs[p])}")

    def test_the_column_order_is_theirs_too(self):
        CE = pytest.importorskip("CEmulator.Emulator",
                                 reason="CEmulator is a [gen] dependency")
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

    def test_it_is_narrower_than_emu_pk_where_it_matters(self):
        """The sigma(M) this recalibrates against comes from `emu_pk`, so the
        two boxes have to be compared rather than assumed compatible."""
        emu_pk_box = pytest.importorskip("emu_pk.box",
                                         reason="emu_pk is the sigma(M) source")
        h_lo = box.BOX["H0"][0] / 100.0
        assert emu_pk_box.BOX["h"][0] < h_lo, "emu_pk should be the wider one in h"
        assert emu_pk_box.BOX["sum_mnu"][1] > box.BOX["mnu"][1]
