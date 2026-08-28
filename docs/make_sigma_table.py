#!/usr/bin/env python
r"""Cache a small :math:`\sigma(M)` table for the documentation's figures.

``emu_hmf`` has no power spectrum and should not acquire one: the caller
supplies :math:`\sigma(M)`, and the released package's two dependencies are
numpy and JAX.  So the one figure that shows an actual abundance needs a
variance from somewhere, and this script produces one -- with CLASS, in the
convention the fit was made in.

Needs ``classy``.  Run it by hand; the table it writes is committed, so the
documentation builds without a Boltzmann solver:

.. code-block:: bash

    python docs/make_sigma_table.py

**The convention matters and is the whole reason this is not a one-liner.**
:math:`\sigma` is of the *cold* field -- baryons plus cold dark matter, with
the massive neutrinos excluded -- against :math:`\bar\rho_{cb}`, because that is
what the correction was fitted against.  Using the total-matter spectrum here
instead would shift the abundance by more than the correction being
illustrated.
"""
import pathlib

import numpy as np

from emu_hmf import box, target

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data" / "sigma_illustrative.npz"

#: Critical density today, in (M_sun/h) / (Mpc/h)^3 -- h-free by construction.
RHO_CRIT_0 = 2.77536627e11

Z = (0.0, 1.0, 2.0)
M = np.logspace(12.0, 14.0, 48)
K = np.logspace(-4.0, np.log10(200.0), 1024)          # h/Mpc


def _window(x):
    """Real-space top hat in Fourier space, and its derivative."""
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    dw = 3.0 * ((x * x - 3.0) * np.sin(x) + 3.0 * x * np.cos(x)) / x ** 4
    return w, dw


def _sigma(r, k, pk):
    r""":math:`(\sigma, \dd\ln\sigma/\dd\ln R)` by quadrature in :math:`\ln k`.

    The logarithmic derivative comes from differentiating the integrand rather
    than from differencing the integral: a finite difference of a quadrature is
    how per-cent noise gets into a mass function.
    """
    x = np.outer(r, k)
    w, dw = _window(x)
    integrand = k ** 3 * pk / (2.0 * np.pi ** 2)
    var = np.trapezoid(integrand * w ** 2, np.log(k), axis=-1)
    dvar = np.trapezoid(integrand * 2.0 * w * dw * x, np.log(k), axis=-1)
    return np.sqrt(var), 0.5 * dvar / var


def main():
    from classy import Class

    theta = dict(zip(box.PARAMS, target.FIDUCIAL))
    h = theta["H0"] / 100.0
    omega_b = theta["Omegab"] * h ** 2
    omega_cdm = (theta["Omegam"] - theta["Omegab"]) * h ** 2   # Omegam is cold
    cosmo = Class()
    cosmo.set({
        "output": "mPk", "P_k_max_h/Mpc": 400.0, "z_max_pk": max(Z) + 0.1,
        "omega_b": omega_b, "omega_cdm": omega_cdm, "h": h,
        "n_s": theta["ns"], "ln10^{10}A_s": float(np.log(1e10 * theta["A"] * 1e-9)),
        "N_ncdm": 1, "m_ncdm": theta["mnu"], "N_ur": 2.0328,
        "Omega_Lambda": 0.0, "w0_fld": theta["w"], "wa_fld": theta["wa"],
        "use_ppf": "yes",
    })
    cosmo.compute()

    rho_cold = theta["Omegam"] * RHO_CRIT_0        # cold, not total
    r = (3.0 * M / (4.0 * np.pi * rho_cold)) ** (1.0 / 3.0)

    sigma, dlns = [], []
    for z in Z:
        # pk_cb_lin wants k in 1/Mpc and returns Mpc^3; convert to h units.
        pk = np.array([cosmo.pk_cb_lin(kk * h, z) * h ** 3 for kk in K])
        s, dlnsdlnr = _sigma(r, K, pk)
        sigma.append(s)
        dlns.append(dlnsdlnr / 3.0)                # d ln sigma / d ln M
    cosmo.struct_cleanup()
    cosmo.empty()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, m=M, z=np.array(Z, dtype=float),
        sigma=np.array(sigma), dlns=np.array(dlns),
        rho_cold=np.float64(rho_cold), theta=target.FIDUCIAL,
        note=np.array("cold field against rho_cb, CLASS linear, illustrative"))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} kB)")
    for z, s in zip(Z, sigma):
        print(f"  z={z:3.1f}  sigma({M[0]:.0e})={s[0]:.3f}  "
              f"sigma({M[-1]:.0e})={s[-1]:.3f}")


if __name__ == "__main__":
    main()
