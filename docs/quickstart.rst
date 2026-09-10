Quickstart
==========

Build a correction, evaluate it, differentiate it.

The multiplicity function
-------------------------

.. code-block:: python

   import numpy as np
   from emu_hmf.model import HmfCorrection

   corr = HmfCorrection()          # the 200m weights, the default

   theta = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.06])
   #                 Omega_b, Omega_cb, H0, n_s, 1e9 A_s, w, w_a, sum m_nu

   f = corr.fsigma(sigma=0.8, theta=theta, z=0.5)

``theta`` is the eight CSST parameters, in the order
:data:`emu_hmf.box.PARAMS`.  Two of them are easy to hand over wrong:

* ``Omega_cb`` is the **cold** density --- baryons plus cold dark matter, with
  the massive neutrinos *excluded*.  It is not the total :math:`\Omega_m`.
* the amplitude is :math:`10^9 A_s`, not :math:`A_s` and not
  :math:`\ln(10^{10}A_s)`.

Either mistake produces a plausible number rather than an error.
:data:`emu_hmf.target.FIDUCIAL` holds a correct point to copy from.

The abundance
-------------

.. code-block:: python

   m = np.logspace(12.0, 14.0, 24)                  # M_sun/h
   sigma, dlnsigma_dlnm, rho_cold = my_variance(m, theta, z=0.5)
   n = corr.dndlnM(m, sigma, dlnsigma_dlnm, rho_cold, theta, z=0.5)

``my_variance`` is the caller's; :doc:`halo_model` says what the three
quantities must be.  The result is

.. math::

   \frac{\dd n}{\dd\ln M} = f(\sigma)\,\frac{\bar\rho_{cb}}{M}\,
       \left|\frac{\dd\ln\sigma}{\dd\ln M}\right| .

The caller supplies :math:`\sigma(M)`; this package computes no power
spectrum.  The variance must be the one the fit was made against, the cold
field against :math:`\bar\rho_{cb}`.  See :doc:`concepts`.

Gradients
---------

The forward pass is JAX throughout, including the cosmology:

.. code-block:: python

   import jax, jax.numpy as jnp
   from emu_hmf import box

   def ln_f(theta):
       return jnp.log(corr.fsigma(0.8, theta, 0.5))

   g = jax.grad(ln_f)(jnp.asarray(theta))       # (8,), one per parameter

``jax.jit`` and ``jax.vmap`` also work; ``vmap`` evaluates a chain of
cosmologies:

.. code-block:: python

   chain = jnp.asarray(box.sample(64))          # or any (n, 8) of cosmologies
   f = jax.jit(lambda t: corr.fsigma(0.8, t, 0.5))
   values = jax.vmap(f)(chain)

Inside a ``jit`` the box check is skipped: the values are not concrete under
tracing.  A jitted forward model is checked once, when it is built.

The virial weights
------------------

.. code-block:: python

   from emu_hmf.model import HmfCorrection, WEIGHTS

   vir = HmfCorrection(WEIGHTS["vir"])

These carry the change of halo boundary as well as the recalibration, so they
sit some 14 per cent below the carrier at :math:`z = 0`.  See :doc:`massdefs`.

When it refuses
---------------

Three things raise.  A cosmology outside the box, below.  A weights file whose
``params_order`` is not the one :func:`~emu_hmf.model.normalise` builds, which
raises when the correction is constructed.  And a ``ggah_mod`` too old for the
conversions, which raises from :func:`~emu_hmf.target.to_ggah_cosmology` when
one is first requested.

.. code-block:: python

   >>> theta_bad = theta.copy(); theta_bad[2] = 55.0      # H0 below the box
   >>> corr.fsigma(0.8, theta_bad, 0.0)
   ValueError: outside the CSST emulator's box, where this recalibration has
   no training data: H0 = 55 not in (60.0, 80.0).  The fit is not defined
   there and will not be extrapolated.

The message names every out-of-bounds parameter.  :doc:`validity` covers the
rest of the domain, including the two bounds that depend on a
:math:`\sigma(M)` this package never sees and the two axes the box does not
carry.
