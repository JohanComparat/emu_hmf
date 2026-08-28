#!/usr/bin/env python
r"""Regenerate every figure in the documentation.

Run by hand, not by the documentation build:

.. code-block:: bash

    python docs/make_figures.py

The PNGs are committed, so ``sphinx-build`` needs numpy, JAX and the shipped
weights and nothing else.  That is the point -- the documentation builder has no
CSST emulator and no Boltzmann solver, and a figure that needed one could not be
rebuilt on it.

The one quantity that genuinely cannot be produced from the released package is
:math:`\sigma(M)`: this package has no power spectrum and should not acquire
one.  A small table of it is cached in ``docs/data/sigma_illustrative.npz`` by
``docs/make_sigma_table.py``, which does need the generation stack, and is
labelled illustrative wherever it is used.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402

from emu_hmf import box, model, target                            # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "_static" / "figures"
SIGMA_TABLE = HERE / "data" / "sigma_illustrative.npz"

# One colour per mass definition, used consistently in every figure.
C = {"200m": "#1f6feb", "vir": "#d1642a"}
LABEL = {"200m": "200m (SO, Rockstar)", "vir": "virial (SO, Rockstar)"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})


def _corr(key):
    return model.HmfCorrection(model.WEIGHTS[key])


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.png")


def fig_correction_vs_nu():
    """What each of the two shipped corrections does to Tinker08."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, z in zip(axes, (0.0, 1.0)):
        lo, hi = target.nu_covered(z)
        nu = np.linspace(lo, hi, 200)
        sigma = target.DELTA_C / nu
        base = np.asarray(model.tinker08(sigma, z))
        for key in ("200m", "vir"):
            r = np.asarray(_corr(key).fsigma(sigma, target.FIDUCIAL, z)) / base
            ax.plot(nu, r, color=C[key], label=LABEL[key])
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax.axvspan(target.NU_TRUSTED[0], lo, color="0.85", zorder=0)
        ax.set_xlabel(r"peak height  $\nu = \delta_c/\sigma$")
        ax.set_title(f"$z = {z:g}$", fontsize=9)
        ax.set_xlim(*target.NU_TRUSTED)
    axes[0].set_ylabel(r"$f_{\rm recal}\,/\,f_{\rm Tinker08}$")
    axes[0].legend(loc="upper left")
    axes[1].text(0.55, 0.06, "shaded: not covered\nat this redshift",
                 transform=axes[1].transAxes, fontsize=7.5, color="0.35")
    fig.suptitle("The correction, relative to the Tinker08 carrier at "
                 r"$\Delta_{\rm m}=200$", fontsize=9.5)
    _save(fig, "correction_vs_nu")


def fig_shape_parameters():
    """Where the recalibration went: amplitude, tilt, pivot, cutoff."""
    z = np.linspace(0.0, 3.0, 120)
    names = [r"$A$ (amplitude)", r"$a$ (low-$\nu$ tilt)",
             r"$b$ (pivot)", r"$c$ (cutoff)"]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.5), sharex=True)
    for key in ("200m", "vir"):
        g = np.asarray(_corr(key).g(target.FIDUCIAL, z))
        for i, ax in enumerate(axes):
            ax.plot(z, np.expm1(g[:, i]) * 100.0, color=C[key],
                    label=LABEL[key] if i == 0 else None)
    for i, ax in enumerate(axes):
        ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
        ax.set_title(names[i], fontsize=9)
        ax.set_xlabel("$z$")
    axes[0].set_ylabel("change (per cent)")
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle("The four Tinker08 shape parameters, as the network moves "
                 "them (zero is the published fit)", fontsize=9.5)
    _save(fig, "shape_parameters")


def fig_cosmology_dependence():
    """The premise: a correction that did not move with cosmology would be a
    constant, and a constant is already inside Tinker08's amplitude."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    lo, hi = target.nu_covered(0.0)
    nu = np.linspace(lo, hi, 200)
    sigma = target.DELTA_C / nu
    base = np.asarray(model.tinker08(sigma, 0.0))
    corr = _corr("200m")
    design = box.sample(6, seed=11)
    cmap = plt.get_cmap("viridis")
    for i, theta in enumerate(design):
        r = np.asarray(corr.fsigma(sigma, theta, 0.0)) / base
        ax.plot(nu, r, color=cmap(i / max(len(design) - 1, 1)), lw=1.2,
                label=r"$\Omega_{cb}=%.2f$, $w=%.2f$, $\Sigma m_\nu=%.2f$"
                      % (theta[1], theta[5], theta[7]))
    r0 = np.asarray(corr.fsigma(sigma, target.FIDUCIAL, 0.0)) / base
    ax.plot(nu, r0, color="k", lw=2.0, label="Planck-like fiducial")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"peak height  $\nu = \delta_c/\sigma$")
    ax.set_ylabel(r"$f_{\rm recal}\,/\,f_{\rm Tinker08}$")
    ax.set_title("The correction moves with the cosmology ($z=0$, 200m)",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, loc="upper left")
    _save(fig, "cosmology_dependence")


def fig_growth_with_redshift():
    """A few per cent at z = 0, and several times that by z = 3."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    zs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    cmap = plt.get_cmap("plasma")
    corr = _corr("200m")
    for i, z in enumerate(zs):
        lo, hi = target.nu_covered(z)
        nu = np.linspace(lo, hi, 150)
        sigma = target.DELTA_C / nu
        r = np.asarray(corr.fsigma(sigma, target.FIDUCIAL, z)) / \
            np.asarray(model.tinker08(sigma, z))
        axes[0].plot(nu, r, color=cmap(i / (len(zs) - 1)), label=f"$z={z:g}$")
    axes[0].axhline(1.0, color="0.4", lw=0.8, ls="--")
    axes[0].set_xlabel(r"$\nu = \delta_c/\sigma$")
    axes[0].set_ylabel(r"$f_{\rm recal}\,/\,f_{\rm Tinker08}$")
    axes[0].legend(fontsize=7, ncol=2)

    zz = np.linspace(0.0, 3.0, 40)
    for key in ("200m", "vir"):
        c = _corr(key)
        size = []
        for z in zz:
            lo, hi = target.nu_covered(z)
            sigma = target.DELTA_C / np.linspace(lo, hi, 80)
            r = np.asarray(c.fsigma(sigma, target.FIDUCIAL, z)) / \
                np.asarray(model.tinker08(sigma, z))
            size.append(np.sqrt(np.mean(np.log(r) ** 2)) * 100.0)
        axes[1].plot(zz, size, color=C[key], label=LABEL[key])
    axes[1].set_xlabel("$z$")
    axes[1].set_ylabel(r"rms of $\ln$ (ratio), per cent")
    axes[1].legend(fontsize=7.5)
    axes[1].set_title("size of the correction", fontsize=9)
    fig.suptitle("The correction grows with redshift, which is why it is a "
                 "function of $z$", fontsize=9.5)
    _save(fig, "growth_with_redshift")


def fig_covered_domain():
    """Where the correction is defined, in the plane a caller thinks in."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    z = np.linspace(0.0, 3.0, 200)
    lo = np.array([target.nu_covered(zz)[0] for zz in z])
    hi = np.array([target.nu_covered(zz)[1] for zz in z])
    ax.fill_between(z, lo, hi, color="#1f6feb", alpha=0.18,
                    label="covered by the training set")
    ax.plot(z, lo, color="#1f6feb", lw=1.4)
    ax.plot(z, hi, color="#1f6feb", lw=1.4)
    ax.axhline(target.NU_TRUSTED[0], color="0.35", ls="--", lw=0.9)
    ax.axhline(target.NU_TRUSTED[1], color="0.35", ls="--", lw=0.9)
    ax.fill_between(z, target.NU_TRUSTED[0], lo, color="0.8", alpha=0.55,
                    label=r"inside $\nu$ range, but not sampled")
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"peak height  $\nu = \delta_c/\sigma$")
    ax.set_ylim(0.3, 3.3)
    ax.set_title(r"The nominal $\nu$ range is necessary, not sufficient",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")
    _save(fig, "covered_domain")


def fig_sensitivity():
    """What a table of numbers cannot give you."""
    corr = model.HmfCorrection()
    lo, hi = target.nu_covered(0.0)
    nu = np.linspace(lo, hi, 60)

    def ln_f(theta, sigma, z):
        return jnp.log(corr.fsigma(sigma, theta, z))

    grad = jax.vmap(jax.grad(ln_f), in_axes=(None, 0, None))
    g = np.asarray(grad(jnp.asarray(target.FIDUCIAL),
                        jnp.asarray(target.DELTA_C / nu), 0.5))
    width = np.array([box.BOX[p][1] - box.BOX[p][0] for p in box.PARAMS])
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(box.PARAMS):
        ax.plot(nu, g[:, i] * width[i], color=cmap(i), label=p)
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"$\nu = \delta_c/\sigma$")
    ax.set_ylabel(r"$\partial \ln f / \partial \theta_i$, per box width")
    ax.set_title(r"Response of $\ln f$ to each parameter ($z=0.5$, 200m), "
                 "by autodiff", fontsize=9.5)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "sensitivity")


def fig_accuracy():
    """Read out of the weights' own metadata, so it cannot drift from them."""
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    keys = ["200m", "vir"]
    y = np.arange(len(keys))
    base = [np.expm1(float(model.load_weights(model.WEIGHTS[k])
                           ["baseline_rms"])) * 100 for k in keys]
    got = [np.expm1(float(model.load_weights(model.WEIGHTS[k])
                          ["val_rms"])) * 100 for k in keys]
    ax.barh(y + 0.18, base, height=0.32, color="0.72", label="Tinker08 unchanged")
    ax.barh(y - 0.18, got, height=0.32, color=[C[k] for k in keys],
            label="recalibrated")
    for i, (b, gv) in enumerate(zip(base, got)):
        ax.text(b + 0.25, i + 0.18, f"{b:.2f} %", va="center", fontsize=8)
        ax.text(gv + 0.25, i - 0.18, f"{gv:.2f} %", va="center", fontsize=8,
                weight="bold")
    ax.set_yticks(y, [LABEL[k] for k in keys], fontsize=8)
    ax.set_xlabel("residual in $\\ln f$ (per cent), held-out cosmologies")
    ax.set_xlim(0, max(base) * 1.25)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "accuracy")


def fig_mass_function():
    """The abundance itself, using the cached illustrative sigma(M)."""
    if not SIGMA_TABLE.exists():
        print(f"  (skipping mass_function: {SIGMA_TABLE.name} not present; "
              "run docs/make_sigma_table.py with the generation stack)")
        return
    with np.load(SIGMA_TABLE) as d:
        m, zs, sigma, dlns, rho = (d["m"], d["z"], d["sigma"], d["dlns"],
                                   float(d["rho_cold"]))
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    cmap = plt.get_cmap("plasma")
    corr = _corr("200m")
    for i, z in enumerate(zs):
        col = cmap(i / max(len(zs) - 1, 1))
        t08 = np.asarray(model.tinker08(sigma[i], z)) * (rho / m) * \
            np.abs(dlns[i])
        rec = np.asarray(corr.dndlnM(m, sigma[i], dlns[i], rho,
                                     target.FIDUCIAL, z))
        # Only the part of the curve the correction was actually fitted on is
        # drawn solid.  At z = 2 a 1e14 halo is nu = 4.3, well past the band --
        # showing that stretch unmarked would be showing an extrapolation as
        # though it were a result.
        lo, hi = target.nu_covered(z)
        nu = target.DELTA_C / sigma[i]
        fitted = (nu >= lo) & (nu <= hi)
        axes[0].loglog(m, t08, color=col, ls="--", lw=1.0)
        axes[0].loglog(m, np.where(fitted, rec, np.nan), color=col, lw=1.8,
                       label=f"$z={z:g}$")
        axes[0].loglog(m, rec, color=col, lw=0.9, alpha=0.35)
        axes[1].semilogx(m, np.where(fitted, rec / t08, np.nan), color=col,
                         lw=1.6)
        axes[1].semilogx(m, rec / t08, color=col, lw=0.9, alpha=0.35)
    axes[1].axhline(1.0, color="0.4", lw=0.8, ls="--")
    axes[0].set_ylabel(r"$\mathrm{d}n/\mathrm{d}\ln M$  $[(h/{\rm Mpc})^3]$")
    axes[1].set_ylabel("recal. / T08")
    axes[1].set_xlabel(r"$M$  $[M_\odot/h]$")
    axes[0].legend(fontsize=7.5)
    axes[0].set_title("Dashed: Tinker08.  Solid: recalibrated (200m).\n"
                      r"Faint: outside the fitted $\nu$ band at that $z$",
                      fontsize=8.5)
    _save(fig, "mass_function")


FIGURES = [fig_correction_vs_nu, fig_shape_parameters, fig_cosmology_dependence,
           fig_growth_with_redshift, fig_covered_domain, fig_sensitivity,
           fig_accuracy, fig_mass_function]

if __name__ == "__main__":
    print(f"writing into {FIG}")
    for f in FIGURES:
        f()
