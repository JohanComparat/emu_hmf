"""Shared fixtures, and the precision the whole suite runs at.

JAX defaults to float32.  Several assertions here are about conversions that
are exact in float64 and good to about 1e-7 in float32 -- a round trip through
a cosmology, a gradient against its closed form -- so a suite that did not pin
the mode would assert one thing on one machine and another thing on the next.
It is pinned here, before anything imports ``jax``, and
``tests/test_public_api.py`` covers the *other* mode in a subprocess, because
a released package must also be correct for a user who never sets the flag.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                        # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

jax.config.update("jax_enable_x64", True)

from emu_hmf import box, target                                   # noqa: E402


@pytest.fixture
def make_shards():
    """Write synthetic shards in the schema :mod:`emu_hmf.generate` writes.

    Enough of a training set to exercise :func:`emu_hmf.fit.load_shards` and
    :func:`emu_hmf.fit.fit` without CLASS, the CSST emulator, or the
    thirty-odd CPU-hours the real campaign costs.
    """
    def _make(directory, n_shards=1, n_cosmo=3, n_z=4, n_m=5,
              massdef=target.DEFAULT_MASSDEF, f=None, sigma=None,
              failed=(), z=None, seed=0):
        directory = str(directory)
        out = []
        for i in range(n_shards):
            theta = np.array([box.sample(1, seed=100 * i + s + seed)[0]
                              for s in range(n_cosmo)])
            zz = np.linspace(0.0, 1.0, n_z) if z is None else np.asarray(z)
            shape = (n_cosmo, n_z, n_m)
            fv = (np.full(shape, 0.2) if f is None
                  else np.broadcast_to(np.asarray(f, float), shape).copy())
            # nu = 1.5 by default: comfortably inside NU_TRUSTED, so nothing is
            # cut unless a test asks for it.
            sg = (np.full(shape, target.DELTA_C / 1.5) if sigma is None
                  else np.broadcast_to(np.asarray(sigma, float), shape).copy())
            path = f"{directory}/hmf_{i:03d}.npz"
            np.savez(path,
                     idx=np.arange(n_cosmo, dtype=np.int64),
                     theta=theta,
                     f=fv.astype(np.float32),
                     sigma=sg.astype(np.float32),
                     dlns=np.full(shape, -0.3, dtype=np.float32),
                     z=zz.astype(np.float64),
                     m=np.logspace(12, 14, n_m),
                     failed_idx=np.array(sorted(failed), dtype=np.int64),
                     massdef=np.array(str(massdef)))
            out.append(path)
        return out
    return _make


@pytest.fixture(scope="module")
def emu():
    """The CSST emulator, with the numpy-2 shim applied.  Skips without it."""
    pytest.importorskip("CEmulator.Emulator",
                        reason="the CSST emulator is not installed")
    pytest.importorskip("ggah_mod.halos._cemulator_compat",
                        reason="the emulator's compatibility shim lives there")
    return target._emulator()
