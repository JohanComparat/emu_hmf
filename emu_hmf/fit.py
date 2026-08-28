r"""Fit the four Tinker08 parameters as functions of cosmology and redshift.

What is minimised is the residual in :math:`\ln f`, over the peak-height range
where the target means something (:data:`emu_hmf.target.NU_TRUSTED`).  In
:math:`\ln` rather than in :math:`f` because :math:`f` spans four decades over
that range and a linear loss would fit the low-:math:`\nu` end and ignore the
clusters.

The network predicts :math:`g`, a log-correction to each of
:math:`(A, a, b, c)`, so :math:`g = 0` is Tinker08 unchanged and the fit starts
there: the output layer is initialised to zero, which means epoch zero is
exactly the published fit and every step after it is a measured improvement on
one.  A fit that cannot get worse than its own starting point is a different
kind of object from one that can.

Needs ``optax``; that is the ``[train]`` extra.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import box, model, target

__all__ = ["Shards", "load_shards", "fit", "main"]


@dataclasses.dataclass(frozen=True)
class Shards:
    """One flat training set, and the record of what it was built from.

    ``provenance`` travels with the arrays rather than beside them.  What the
    weights were fitted on is part of what the weights mean, and it is written
    into the ``.npz`` at the end of :func:`fit`, so whatever quotes the
    accuracy later can read the sample size out of the same file.
    """

    theta: np.ndarray          #: ``(n_rows, 8)`` in :data:`emu_hmf.box.PARAMS` order
    z: np.ndarray              #: ``(n_rows,)``
    sigma: np.ndarray          #: ``(n_rows,)``
    ln_f: np.ndarray           #: ``(n_rows,)`` -- what is fitted
    cosmo_id: np.ndarray       #: ``(n_rows,)`` which design each row came from
    provenance: dict


def load_shards(shard_dir, nu_range=None) -> "Shards":
    r"""Every shard in ``shard_dir`` -> one :class:`Shards`, flat over (cosmology, z, M).

    Rows outside ``nu_range`` are dropped here rather than weighted down: a
    weight of zero and an absent row differ only in how long the optimiser
    spends on them, and the reason for the cut is that the target is not
    trustworthy there, which is not a statement about weighting.
    """
    lo, hi = target.NU_TRUSTED if nu_range is None else nu_range
    files = sorted(pathlib.Path(shard_dir).glob("hmf_*.npz"))
    if not files:
        raise FileNotFoundError(f"no hmf_*.npz under {shard_dir}")

    # One definition per fit.  The correction is a correction *to tinker08 at a
    # particular Delta*, so averaging shards from two definitions would fit a
    # correction for neither -- and every number in the result would look
    # entirely reasonable.
    defs = set()
    for f in files:
        with np.load(f) as d:
            if "massdef" not in d:
                raise ValueError(
                    f"{f.name} does not say which halo definition it holds.  "
                    "A shard that cannot name its own mass definition cannot "
                    "be fitted, because the correction is a correction to "
                    "tinker08 at a particular Delta.")
            defs.add(str(d["massdef"]))
    if len(defs) > 1:
        raise ValueError(
            f"{shard_dir} mixes halo definitions {sorted(defs)}.  A correction "
            "is fitted to one of them; fit them separately and compare the "
            "results, which is the only way to learn whether they agree.")
    massdef = defs.pop()
    sig, lnf, ths, zs, cid = [], [], [], [], []
    n_cosmo = n_failed = 0
    for f in files:
        with np.load(f) as d:
            theta, fv, sg, z = d["theta"], d["f"], d["sigma"], d["z"]
            n_cosmo += len(theta)
            n_failed += len(d["failed_idx"])
        n_c, n_z, n_m = fv.shape
        th = np.repeat(theta[:, None, :], n_z, axis=1)         # (n_c, n_z, 8)
        zz = np.broadcast_to(z[None, :], (n_c, n_z))
        nu = target.DELTA_C / sg
        ok = np.isfinite(fv) & (fv > 0) & (nu >= lo) & (nu <= hi)
        ic, iz, im = np.nonzero(ok)
        ths.append(th[ic, iz])
        zs.append(zz[ic, iz])
        sig.append(sg[ic, iz, im])
        lnf.append(np.log(fv[ic, iz, im]))
        # Which design each row came from, offset so the ids stay unique
        # across shards.  `fit` splits on this and not on the row index.
        cid.append(ic + (n_cosmo - len(theta)))
    theta = np.concatenate(ths).astype(np.float64)
    z = np.concatenate(zs).astype(np.float64)
    sigma = np.concatenate(sig).astype(np.float64)
    ln_f = np.concatenate(lnf).astype(np.float64)
    cosmo_id = np.concatenate(cid).astype(np.int64)
    print(f"{len(files)} shards of {massdef}, {n_cosmo} cosmologies "
          f"({n_failed} refused) "
          f"-> {len(ln_f)} rows in nu = [{lo}, {hi}]", flush=True)
    return Shards(theta=theta, z=z, sigma=sigma, ln_f=ln_f,
                  cosmo_id=cosmo_id,
                  provenance={"massdef": massdef, "n_shards": len(files),
                              "n_cosmologies": n_cosmo, "n_refused": n_failed,
                              "n_rows": int(len(theta)),
                              "nu_lo": float(lo), "nu_hi": float(hi)})


def _init(key, sizes, scale=0.1):
    """Zero output layer: epoch zero *is* Tinker08."""
    p, n = {}, len(sizes) - 1
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k1 = jax.random.split(key)
        w = jax.random.normal(k1, (a, b)) * (scale * np.sqrt(2.0 / a))
        p[f"W{i}"] = jnp.zeros((a, b)) if i == n - 1 else w
        p[f"b{i}"] = jnp.zeros(b)
    return p


def fit(shard_dir, out, hidden=(64, 64), epochs=400, batch=4096, lr=3e-3,
        seed=0, val_frac=0.1, nu_range=None):
    # Float64, always.  A network fitted in single precision and one fitted in
    # double are two different sets of weights, and which one a run produced
    # would otherwise depend on an environment variable set somewhere else.
    # The residual being measured is half a per cent; the precision it is
    # measured at is not a detail to leave to the caller.
    jax.config.update("jax_enable_x64", True)
    import optax

    d = load_shards(shard_dir, nu_range)
    theta, z, sigma, ln_f, cosmo_id = (d.theta, d.z, d.sigma, d.ln_f,
                                       d.cosmo_id)
    x = np.asarray(model.normalise(theta, z))
    assert x.shape == (len(theta), len(box.PARAMS) + 1), x.shape

    # Split on *cosmologies*, not on rows.  Each design contributes a few
    # hundred rows -- twelve redshifts times the masses inside the peak-height
    # cut -- and those rows are not independent: at fixed cosmology, ln f is a
    # smooth function of sigma, so a random row split leaves the network
    # interpolating between neighbouring masses of a design it has already
    # seen.  That measures something real, but it is not generalisation to a
    # new cosmology, which is the only thing this correction is ever asked
    # for.  Measured on the first full run, the difference is not academic:
    # the row split reported 0.59 per cent.
    rng = np.random.default_rng(seed)
    ids = np.unique(cosmo_id)
    n_held = max(1, int(round(len(ids) * val_frac)))
    held = set(rng.permutation(ids)[:n_held].tolist())
    in_val = np.isin(cosmo_id, list(held))
    val, tr = np.nonzero(in_val)[0], np.nonzero(~in_val)[0]
    print(f"held out {len(held)} of {len(ids)} cosmologies entirely "
          f"({len(val)} of {len(cosmo_id)} rows)", flush=True)

    sizes = [x.shape[1], *hidden, 4]
    params = _init(jax.random.PRNGKey(seed), sizes)
    opt = optax.adam(lr)
    state = opt.init(params)

    X = jnp.asarray(x); S = jnp.asarray(sigma); Z = jnp.asarray(z)
    Y = jnp.asarray(ln_f)

    def loss(p, idx):
        g = model._mlp(p, X[idx])
        pred = jnp.log(model.tinker08(S[idx], Z[idx], g))
        return jnp.mean((pred - Y[idx]) ** 2)

    @jax.jit
    def step(p, s, idx):
        l, grad = jax.value_and_grad(loss)(p, idx)
        upd, s = opt.update(grad, s)
        return optax.apply_updates(p, upd), s, l

    base = float(np.sqrt(np.mean(
        (np.asarray(jnp.log(model.tinker08(S[val], Z[val]))) - ln_f[val]) ** 2)))
    print(f"Tinker08 unchanged, on the held-out split: rms {base:.5f} in ln f "
          f"({np.expm1(base):.2%})", flush=True)

    tr_j = jnp.asarray(tr)
    best, best_p = float("inf"), params
    n_batch = max(1, len(tr) // batch)
    for ep in range(epochs):
        t0 = time.time()
        order = jnp.asarray(rng.permutation(len(tr)))
        tot = 0.0
        for b in range(n_batch):
            idx = tr_j[order[b * batch:(b + 1) * batch]]
            params, state, l = step(params, state, idx)
            tot += float(l)
        vl = float(loss(params, jnp.asarray(val)))
        if vl < best:
            best, best_p = vl, params
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  epoch {ep + 1:4d}/{epochs}  train {tot / n_batch:.6f}  "
                  f"val {vl:.6f}  rms {np.sqrt(vl):.5f}  "
                  f"{time.time() - t0:.1f} s", flush=True)

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prov = dict(d.provenance)
    # Recorded so the number the paper quotes cannot be mistaken for the
    # optimistic one a row split would give.
    prov["n_val_cosmologies"] = int(len(held))
    prov["split_by_cosmology"] = 1
    np.savez(out, **{k: np.asarray(v) for k, v in best_p.items()},
             n_layers=np.int64(len(sizes) - 1),
             val_rms=np.float64(np.sqrt(best)),
             baseline_rms=np.float64(base),
             nu_range=np.array(target.NU_TRUSTED if nu_range is None else nu_range),
             params_order=np.array(list(box.PARAMS) + ["z"], dtype="U16"),
             epochs=np.int64(epochs), hidden=np.asarray(hidden, dtype=np.int64),
             val_frac=np.float64(val_frac),
             **{k: (np.array(v) if isinstance(v, str)
                    else np.int64(v) if isinstance(v, int)
                    else np.float64(v))
                for k, v in prov.items()})
    print(f"\nwrote {out}", flush=True)
    print(f"  Tinker08 unchanged : rms {base:.5f} in ln f  ({np.expm1(base):.2%})")
    print(f"  recalibrated       : rms {np.sqrt(best):.5f} in ln f  "
          f"({np.expm1(np.sqrt(best)):.2%})")
    print(f"  improvement        : {base / np.sqrt(best):.2f}x")
    print(f"  fitted on          : {prov.get('n_cosmologies', '?')} cosmologies, "
          f"{prov.get('n_rows', '?')} rows, {prov.get('n_refused', '?')} refused")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default="emu_hmf/data/emu_hmf_mlp.npz")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, nargs="*", default=[64, 64])
    a = ap.parse_args(argv)
    fit(a.shards, a.out, hidden=tuple(a.hidden), epochs=a.epochs)


if __name__ == "__main__":       # pragma: no cover
    main()
