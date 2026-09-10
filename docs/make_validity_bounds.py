r"""Measure what the eight parameters cannot represent, and print the tables.

Two axes reach a mass function that :data:`emu_hmf.box.PARAMS` has no column
for: the curvature :math:`\Omega_k` and the neutrino mass ordering.  Neither can
be trained --- the CSST emulator's box is flat and its neutrinos are one
species --- so what this package owes its callers is a *number* rather than a
refusal, and this is the script that measures it.

Run from the repository root, in an environment with CSSTemu and CLASS::

    python docs/make_validity_bounds.py

It writes ``docs/data/validity_bounds.npz`` and prints the two reStructuredText
tables that go into ``docs/validity.rst``.  Same shape as
``docs/make_sigma_table.py``: the numbers in the documentation are generated,
not typed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emu_hmf import box, generate, model, target        # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "data" / "validity_bounds.npz"

#: The largest :math:`|\Omega_k|` the quoted coefficient is meaned over.
#: Beyond it the sweep still runs --- that is how the fall is measured --- but
#: those points describe the tail rather than the number callers use.
QUOTED_TO = 0.05

#: Curvatures to perturb by.  Spanning a factor 25 so linearity is *measured*
#: rather than assumed --- a coefficient quoted from one amplitude is a slope
#: through one point.
OMEGA_K = (0.002, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45)

#: Where the ordering measurement is dense.  Normal ordering has no solution
#: below 0.058993 eV and the whole effect lives in the first 0.02 eV above it,
#: so a uniform grid over the box would spend every point where there is
#: nothing to see.
def _mnu_grid():
    from ggah_mod.cosmology import constants as C
    floor_n = C.NU_MASS_FLOOR["normal"]
    floor_i = C.NU_MASS_FLOOR["inverted"]
    return floor_n, floor_i, np.array(
        [floor_n, 0.0595, 0.060, 0.065, 0.070, 0.080,
         0.100, 0.125, 0.150, 0.200, 0.250, 0.300])


# --------------------------------------------------------------- curvature
def curvature_cost(massdef, thetas, omk_values=OMEGA_K):
    r"""``{|Omega_k|: max |d ln f|}`` from perturbing the target's own carrier.

    CSSTemu's ``dn/dlnM`` is a GP-emulated ratio times a Castro+23 baseline
    whose entire explicit cosmology dependence at fixed
    :math:`(\sigma, \dd\ln\sigma/\dd\ln R)` is :math:`\Omega_m(z)` and
    :math:`\delta_c(\Omega_m(z))` --- and ``get_Ez`` already carries
    :math:`\Omega_k(1+z)^2`, which ``set_cosmos`` merely hard-sets to zero.  So
    the curvature response is reachable by setting it back.

    :math:`\sigma` comes from the emulator's *flat-trained* ``P_cb``, which
    knows nothing about :math:`\Omega_k`, so the comparison is automatically at
    fixed :math:`\sigma` --- which is the quantity wanted, because this package
    takes :math:`\sigma(M)` as an input and curvature reaches it there.

    **Self-validating.**  At :math:`z = 0`, :math:`E(0) = 1` by closure whatever
    :math:`\Omega_k` is, so :math:`\Delta\ln f(z{=}0)` must be identically zero.
    Getting the ``OmegaL`` closure wrong is the one way to botch this recipe,
    and that identity catches it.
    """
    emu = target._emulator()
    z = np.asarray(target.Z_TRAINED, dtype=float)
    m = generate.M_GRID
    out = {k: 0.0 for k in omk_values}
    z0 = 0.0

    for th in thetas:
        target.set_cosmology(emu, th)
        C = emu.Cosmo
        base = np.log(np.asarray(emu.get_dndlnM_Castro23(
            z=z, M=m, Pcb=True, revisted=True, massdef=massdef)))
        keep = np.isfinite(base)
        for omk in omk_values:
            target.set_cosmology(emu, th)          # reset every time
            C = emu.Cosmo
            C.Omegak = omk
            # The closure line in CSSTemu omits Omegak; restore it by hand or
            # the perturbation silently changes the dark energy instead.
            C.OmegaL = 1 - C.Omegam - C.Omeganu - C.OmegaR - omk
            pert = np.log(np.asarray(emu.get_dndlnM_Castro23(
                z=z, M=m, Pcb=True, revisted=True, massdef=massdef)))
            d = np.abs(pert - base)
            ok = keep & np.isfinite(pert)
            out[omk] = max(out[omk], float(d[ok].max()))
            z0 = max(z0, float(np.abs(d[0][ok[0]]).max()))
    return out, z0


# ---------------------------------------------------------------- ordering
def ordering_cost(thetas, hierarchy="normal"):
    r"""``{sum_mnu: (max, rms) |d ln f|}`` between an ordering and degenerate.

    :func:`emu_hmf.target.theta_from_cosmology` returns eight
    hierarchy-invariant numbers, so :math:`g(\theta, z)` is *provably* unchanged
    by the split.  What moves is the target the shipped :math:`g` was fitted
    against: :math:`\bar\rho_{cb}` is bit-identical, so the whole effect is
    :math:`\sigma(M)` and :math:`\dd\ln\sigma/\dd\ln M`, through

    .. math::  \ln f = {\rm const}(M) - \ln|\dd\ln\sigma/\dd\ln M|

    CLASS, because it takes the three masses as an explicit list.  ``emu_pk``
    2.0.0 carries them too, as the two ratios ``nu_r1`` and ``nu_r2``, but it
    reaches them through a fitted network where CLASS solves them, and this is
    the measurement the network's own accuracy would otherwise be folded into.
    """
    import jax.numpy as jnp
    from ggah_mod.cosmology import Cosmology
    from ggah_mod.cosmology.power import make_pk
    from ggah_mod.halos.variance import dln_sigma_dln_mass, sigma_of_mass

    pk = make_pk("class")
    z = np.asarray(target.Z_TRAINED, dtype=float)
    m = generate.M_GRID
    k = generate.K_GRID
    lo, hi = target.NU_TRUSTED
    _, _, grid = _mnu_grid()
    res = {}

    def chain(cosmo):
        p = np.asarray(pk.pk_cb(k, z, cosmo))
        rho = float(cosmo.rho_cold)
        s = np.stack([np.asarray(sigma_of_mass(jnp.asarray(m), k, pp, rho))
                      for pp in p])
        d = np.stack([np.asarray(dln_sigma_dln_mass(jnp.asarray(m), k, pp, rho))
                      for pp in p])
        return s, d

    for mnu in grid:
        worst, acc = 0.0, []
        for th in thetas:
            t = np.asarray(th, dtype=float).copy()
            t[box.PARAMS.index("mnu")] = mnu
            deg = target.to_ggah_cosmology(t)
            try:
                oth = deg.replace(nu_hierarchy=hierarchy)
            except ValueError:
                continue                    # below this ordering's floor
            s_d, d_d = chain(deg)
            s_o, d_o = chain(oth)
            nu = target.DELTA_C / s_d
            ok = (nu >= lo) & (nu <= hi) & np.isfinite(d_d) & np.isfinite(d_o)
            if not ok.any():
                continue
            # ln f at fixed sigma: the const(M) cancels, the Jacobian does not.
            dlnf = np.abs(np.log(np.abs(d_o)) - np.log(np.abs(d_d)))[ok]
            worst = max(worst, float(dlnf.max()))
            acc.append(dlnf)
        if acc:
            flat = np.concatenate(acc)
            res[float(mnu)] = (worst, float(np.sqrt(np.mean(flat ** 2))))
    return res


# ------------------------------------------------------------------ report
def measured_crossing(cost, val_rms, baseline_rms):
    r"""Where the sweep *observes* the correction stop beating its own carrier.

    ``None`` when the sweep never gets there --- the correction is still ahead
    at the largest curvature measured, so the crossing is beyond the range and
    any number for it is an extrapolation.

    Returned beside the linear-law crossover rather than instead of it, because
    the two disagreeing by a known amount in a known direction is the whole
    defence of quoting a falling coefficient linearly.
    """
    ks = sorted(cost)
    eff = [np.sqrt(val_rms ** 2 + cost[k] ** 2) for k in ks]
    for i in range(len(ks) - 1):
        if eff[i] < baseline_rms <= eff[i + 1]:
            f = (baseline_rms - eff[i]) / (eff[i + 1] - eff[i])
            return float(ks[i] + f * (ks[i + 1] - ks[i]))
    return None


def _rst_curvature(cost):
    """The table `docs/validity.rst` publishes, not a different one.

    It used to print the *parity* curvature --- where the induced error equals
    the residual --- under a heading that read like the crossover, which is
    where it equals the ``tinker08`` baseline instead.  The two differ by a
    factor of thirteen at 200m, and only the second decides anything.
    """
    lines = [".. list-table::", "   :header-rows: 1",
             "   :widths: 20 26 26 28", "",
             "   * - weights", "     - linear crossover",
             "     - observed crossing", "     - the law is early by"]
    for key, (c, crossover, observed) in cost.items():
        early = "n/a" if observed is None else f"{1 - crossover / observed:.0%}"
        lines += [f"   * - ``{key}``", f"     - {crossover:.4f}",
                  f"     - {'not reached' if observed is None else f'{observed:.4f}'}",
                  f"     - {early}"]
    return "\n".join(lines)


def _rst_ordering(res):
    lines = [".. list-table::", "   :header-rows: 1", "   :widths: 34 33 33", "",
             "   * - Sum m_nu [eV]", "     - max |d ln f|", "     - rms |d ln f|"]
    for mnu in sorted(res):
        mx, rms = res[mnu]
        lines += [f"   * - {mnu:.4f}", f"     - {mx:.2e}", f"     - {rms:.2e}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curvature-only", action="store_true",
                    help="skip the CLASS half and merge into the existing "
                         "archive.  The ordering sweep costs a Boltzmann solve "
                         "per point and does not move when the curvature "
                         "measurement is re-run.")
    a = ap.parse_args(argv)
    thetas = box.sample(40, seed=11)
    fid = target.FIDUCIAL

    print("== curvature ==", flush=True)
    curv, saved = {}, {}
    for key, md in (("200m", "RockstarM200m"), ("vir", "RockstarMvir")):
        cost, z0 = curvature_cost(md, thetas)
        coeffs = {k: v / k for k, v in cost.items()}
        # The coefficient is the SMALL-curvature one, meaned over the points at
        # or below `QUOTED_TO`, because that is the regime it is quoted in --
        # realistic priors sit two orders below the far end of this sweep.
        #
        # Meaning it over the whole sweep instead would dilute it with the
        # falling tail and hand back a number that is wrong where it is used
        # and only right on average.  The tail is not discarded: it is what
        # `measured_to` and the linearity table are for, and what says the
        # linear reading is conservative.
        c = float(np.mean([v for k, v in coeffs.items() if k <= QUOTED_TO]))
        spread = (max(coeffs.values()) - min(coeffs.values())) / c
        with np.load(model.WEIGHTS[key]) as w:
            rms, base = float(w["val_rms"]), float(w["baseline_rms"])
        obs = measured_crossing(cost, rms, base)
        crossover = float(np.sqrt(base ** 2 - rms ** 2) / c)
        curv[key] = (c, crossover, obs)
        saved[f"curv_{key}"] = np.array([cost[k] for k in OMEGA_K])
        saved[f"crossing_{key}"] = obs
        print(f"  {md:15s} coefficient {c:.4f} (|Omega_k| <= {QUOTED_TO})"
              f"  spread over the full sweep {spread:.1%}"
              f"  z=0 identity {z0:.3e}  parity {rms / c:.4f}"
              f"  observed crossing {obs}", flush=True)
        assert z0 < 1e-12, f"the z=0 identity failed: {z0}"

    if a.curvature_only and OUT.exists():
        with np.load(OUT) as prev:
            order = {float(m): (float(x), float(r)) for m, x, r
                     in zip(prev["mnu"], prev["order_max"], prev["order_rms"])}
        print("\n== neutrino ordering: kept from the existing archive ==",
              flush=True)
    else:
        print("\n== neutrino ordering ==", flush=True)
        order = ordering_cost([fid] + list(thetas[:4]), "normal")
    for mnu in sorted(order):
        print(f"  Sum m_nu = {mnu:.4f}  max {order[mnu][0]:.3e}"
              f"  rms {order[mnu][1]:.3e}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, omega_k=np.array(OMEGA_K),
             coeff_200m=np.float64(curv["200m"][0]),
             coeff_vir=np.float64(curv["vir"][0]),
             measured_to=np.float64(max(OMEGA_K)),
             quoted_to=np.float64(QUOTED_TO),
             crossing_200m=np.float64(
                 np.nan if saved["crossing_200m"] is None
                 else saved["crossing_200m"]),
             crossing_vir=np.float64(
                 np.nan if saved["crossing_vir"] is None
                 else saved["crossing_vir"]),
             mnu=np.array(sorted(order)),
             order_max=np.array([order[m][0] for m in sorted(order)]),
             order_rms=np.array([order[m][1] for m in sorted(order)]),
             **{k: v for k, v in saved.items() if k.startswith("curv_")})
    print(f"\nwrote {OUT}\n")
    print(_rst_curvature(curv), "\n")
    print(_rst_ordering(order))


if __name__ == "__main__":       # pragma: no cover
    main()
