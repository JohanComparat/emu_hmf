r"""Build the training set: the CSST mass function as a multiplicity function.

One row is one :math:`(\theta, z, M)`.  What is stored is not
:math:`\dd n/\dd\ln M` but

.. math::

    f_{\rm target}(\sigma) = \frac{\dd n}{\dd\ln M}\,
        \frac{M}{\bar\rho_{cb}\,|\dd\ln\sigma/\dd\ln M|}

--- the emulator's abundance converted into a multiplicity function *in this
package's variance convention*.  That conversion is the whole design.  Fitting
:math:`f(\sigma)` against one :math:`\sigma(M)` and evaluating it with another
is the mismatch that makes a multiplicity function look wrong when the
convention around it is what moved, and doing the conversion here means the fit
absorbs it once, at generation, rather than leaving it to every caller.

**Why CLASS and not** :mod:`emu_pk`.  The variance has to be available
everywhere the CSST box goes, and :mod:`emu_pk` was trained on
:math:`\omega_b \in [0.017, 0.028]` while CSST reaches
:math:`0.0145` to :math:`0.0382` --- so a network spectrum covers only 70 per
cent of the box being recalibrated, and the missing 30 per cent is not a corner
but a slab.  CLASS has no box.  Generation is offline and one-off, so paying
3.6 s a cosmology to remove an avoidable approximation from the training data is
the easy side of that trade.

Shards write every ``CHUNK`` cosmologies and skip what is already on disk, for
the reason :mod:`emu_pk.generate` gives: a besteffort kill that loses an hour of
completed solves is a lesson one pays for once.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from . import box, target

__all__ = ["M_GRID", "CHUNK", "solve_one", "shard", "main"]

#: Where the target is smooth enough to fit; see :data:`emu_hmf.target.M_TRUSTED`.
M_GRID = np.logspace(np.log10(target.M_TRUSTED[0]),
                     np.log10(target.M_TRUSTED[1]), 24)

#: Cosmologies per write.  Small enough that a kill costs minutes.
CHUNK = 25

#: The wavenumbers sigma(M) is integrated over.  Wider than the masses need,
#: because the top-hat window has no sharp edge -- see the guard in
#: ``ggah_mod.halos.variance.check_k_support``, which refuses a grid too narrow
#: for its mass range and which this grid clears by more than six times.
K_GRID = np.logspace(-4.0, np.log10(200.0), 512)


def solve_one(theta, emu, pk, z=None, m=None):
    r"""``(f_target, sigma, dlns)`` for one cosmology, shape ``(n_z, n_m)``.

    Raises whatever CLASS or the emulator raise: a refusal is data, and
    :func:`shard` records which design index produced it rather than dropping it
    silently.
    """
    import jax.numpy as jnp
    from ggah_mod.halos.variance import sigma_of_mass, dln_sigma_dln_mass

    z = np.asarray(target.Z_TRAINED if z is None else z, dtype=float)
    m = M_GRID if m is None else np.asarray(m, dtype=float)

    cosmo = target.to_ggah_cosmology(theta)
    p_cb = np.asarray(pk.pk_cb(K_GRID, z, cosmo))       # (n_z, n_k)
    rho = float(cosmo.rho_cold)

    sig = np.stack([np.asarray(sigma_of_mass(jnp.asarray(m), K_GRID, p, rho))
                    for p in p_cb])
    dlns = np.stack([np.asarray(dln_sigma_dln_mass(jnp.asarray(m), K_GRID, p, rho))
                     for p in p_cb])

    dndlnm = np.asarray(target.csst_dndlnM(theta, z, m, emu=emu))
    dndlnm = dndlnm.reshape(len(z), len(m))
    f = dndlnm * m[None, :] / (rho * np.abs(dlns))
    return f, sig, dlns


def shard(index: int, n_per_shard: int, out, n_total: int,
          seed: int = 20260828, chunk: int = CHUNK):
    """Solve one contiguous slice of the design and write it.

    The design is regenerated from the seed rather than read from a file, so a
    shard can be rebuilt later without shipping the matrix and two shards can
    never disagree about which index means which cosmology.
    """
    out = pathlib.Path(out)
    if out.exists():
        print(f"{out.name} exists; skipping", flush=True)
        return out

    from ggah_mod.cosmology.power import make_pk
    design = box.sample(n_total, seed=seed)
    lo = index * n_per_shard
    hi = min(lo + n_per_shard, n_total)
    emu = target._emulator()
    pk = make_pk("class")

    idx, thetas, fs, sigs, dlnss, failed = [], [], [], [], [], []
    t0 = time.monotonic()
    for j in range(lo, hi):
        try:
            f, sig, dlns = solve_one(design[j], emu, pk)
        except Exception as exc:                      # noqa: BLE001
            failed.append(j)
            print(f"  design {j}: {type(exc).__name__}: {str(exc)[:70]}",
                  flush=True)
            continue
        idx.append(j)
        thetas.append(design[j])
        fs.append(f)
        sigs.append(sig)
        dlnss.append(dlns)
        done = j - lo + 1
        if done % chunk == 0 or j == hi - 1:
            _write(out, idx, thetas, fs, sigs, dlnss, failed)
            dt = time.monotonic() - t0
            print(f"  {done}/{hi - lo}  {dt:6.1f} s  "
                  f"(eta {dt * (hi - lo - done) / done:6.1f} s)", flush=True)
    return out


def _write(out, idx, thetas, fs, sigs, dlnss, failed):
    tmp = out.with_name(out.stem + ".part.npz")       # `savez` appends `.npz`
    np.savez_compressed(
        tmp,
        idx=np.array(idx, dtype=np.int64),
        theta=np.array(thetas, dtype=np.float64).reshape(len(idx), -1),
        f=np.array(fs, dtype=np.float32),
        sigma=np.array(sigs, dtype=np.float32),
        dlns=np.array(dlnss, dtype=np.float32),
        z=np.array(target.Z_TRAINED, dtype=np.float64),
        m=M_GRID.astype(np.float64),
        failed_idx=np.array(sorted(set(failed)), dtype=np.int64),
    )
    tmp.rename(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-per-shard", type=int, default=100)
    ap.add_argument("--n-total", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    shard(a.shard, a.n_per_shard, a.out, a.n_total, a.seed)


if __name__ == "__main__":       # pragma: no cover
    main()
