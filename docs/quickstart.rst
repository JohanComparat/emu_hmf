Quickstart
==========

Build a correction, evaluate it, differentiate it.

The multiplicity function
-------------------------

.. code-block:: python

   import numpy as np
   from emu_hmf.model import HmfCorrection

   corr = HmfCorrection()          # the 200m weights, which is the default

   theta = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.06])
   #                 Omega_b, Omega_cb, H0, n_s, 1e9 A_s, w, w_a, sum m_nu

   f = corr.fsigma(sigma=0.8, theta=theta, z=0.5)

``theta`` is the eight CSST parameters, in the order
:data:`emu_hmf.box.PARAMS`.  Two of them are easy to hand over wrong:

* ``Omega_cb`` is the **cold** density --- baryons plus cold dark matter, with
  the massive neutrinos *excluded*.  It is not the total :math:`\Omega_m`.
* the amplitude is :math:`10^9 A_s`, not :math:`A_s` and not
  :math:`\ln(10^{10}A_s)`.

Getting either wrong produces a plausible number rather than an error, which is
why :data:`emu_hmf.target.FIDUCIAL` exists to copy from.

The abundance
-------------

.. code-block:: python

   n = corr.dndlnM(m, sigma, dlnsigma_dlnm, rho_cold, theta, z=0.5)

which is

.. math::

   \frac{\dd n}{\dd\ln M} = f(\sigma)\,\frac{\bar\rho_{cb}}{M}\,
       \left|\frac{\dd\ln\sigma}{\dd\ln M}\right| .

:math:`\sigma(M)` is passed in and not computed.  This package has no power
spectrum and should not acquire one --- but the variance it is handed has to be
the one it was fitted against: the **cold** field against
:math:`\bar\rho_{cb}`.  Fitting :math:`f(\sigma)` against one variance and
evaluating it with another is the mismatch that makes a multiplicity function
look wrong when the convention around it is what moved.  See :doc:`concepts`.

Gradients
---------

Everything is JAX, all the way through the cosmology:

.. code-block:: python

   import jax, jax.numpy as jnp

   def ln_f(theta):
       return jnp.log(corr.fsigma(0.8, theta, 0.5))

   g = jax.grad(ln_f)(jnp.asarray(theta))       # (8,), one per parameter

``jax.jit`` and ``jax.vmap`` work too --- the latter is how a chain of
cosmologies is evaluated:

.. code-block:: python

   f = jax.jit(lambda t: corr.fsigma(0.8, t, 0.5))
   values = jax.vmap(f)(chain)                  # chain is (n, 8)

Inside a ``jit`` the box check is skipped, because the values are not available
under tracing and raising there would break the gradient the package exists to
provide.  A jitted forward model is checked once, when it is built.

The virial weights
------------------

.. code-block:: python

   from emu_hmf.model import HmfCorrection, WEIGHTS

   vir = HmfCorrection(WEIGHTS["vir"])

These are **not** a per-cent correction: they carry the change of halo boundary
as well as the recalibration.  :doc:`massdefs` says what that means and why the
two are separate files.

When it refuses
---------------

.. code-block:: python

   >>> theta_bad = theta.copy(); theta_bad[2] = 55.0      # H0 below the box
   >>> corr.fsigma(0.8, theta_bad, 0.0)
   ValueError: outside the CSST emulator's box, where this recalibration has
   no training data: H0 = 55 not in (60.0, 80.0).  The fit is not defined
   there and will not be extrapolated.

Every out-of-bounds parameter is named, not just the first.  See
:doc:`validity` for the second half of the domain --- the one that is *not*
checked for you, because it depends on a :math:`\sigma(M)` this package never
sees.
