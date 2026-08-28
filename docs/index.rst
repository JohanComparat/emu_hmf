emu_hmf
=======

A differentiable, cosmology-dependent recalibration of the Tinker et al. (2008)
halo multiplicity function, trained against the CSST emulator over the box that
emulator was built on.

Tinker08 is a fit to simulations, and not to the simulations anyone compares
against now: at a Planck cosmology it is offset by a few per cent at
:math:`z = 0`, and the size of that offset is itself a function of cosmology and
redshift, which a fit whose only inputs are :math:`\sigma(M)` and :math:`z`
cannot express.  This package learns that offset --- as a correction to
Tinker08's own four shape parameters, so that at zero correction the answer *is*
Tinker08, exactly.

.. code-block:: bash

   pip install emu_hmf

Two dependencies, numpy and JAX.  ``jax.grad``, ``jax.jit`` and ``jax.vmap`` all
pass through the cosmology, which is the reason this is a package and not a
table of numbers.

.. figure:: _static/figures/correction_vs_nu.png
   :width: 100%
   :align: center
   :alt: the correction relative to Tinker08, at two halo definitions

   What the two shipped corrections do to Tinker08, at the Planck-like
   fiducial.  See :doc:`massdefs` for why there are two and not one.

.. toctree::
   :maxdepth: 2
   :caption: Using it

   installation
   quickstart
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: What it is

   concepts
   massdefs
   validity

.. toctree::
   :maxdepth: 2
   :caption: Beyond the package

   halo_model
   reproducing
   testing
   citation

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/model
   api/box
   api/target
   api/generate
   api/fit

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
