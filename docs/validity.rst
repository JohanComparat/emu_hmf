Where the correction is defined
===============================

This package refuses a cosmology or a mass outside the ranges it was fitted
over.  Two further quantities affect the answer without appearing in the box,
and it records their cost instead.

The cosmology: checked
----------------------

The eight parameters must be inside the CSST emulator's box.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - :math:`\Omega_b`
     - :math:`\Omega_{cb}`
     - :math:`H_0`
     - :math:`n_s`
     - :math:`10^9 A_s`
   * - 0.04 – 0.06
     - 0.24 – 0.40
     - 60 – 80
     - 0.92 – 1.00
     - 1.7 – 2.5

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - :math:`w_0`
     - :math:`w_a`
     - :math:`\Sigma m_\nu` [eV]
   * - −1.3 – −0.7
     - −0.5 – 0.5
     - 0 – 0.3

:math:`\Omega_{cb}` is the cold density, with massive neutrinos excluded.

:mod:`emu_hmf.box` copies these numbers rather than importing them, so that a
forecast need not install a Gaussian-process emulator to read its own bounds.
``tests/test_box.py`` asserts the copy against the emulator's ``param_limits``.

A cosmology outside the box raises, naming every offending parameter:

.. code-block:: python

   >>> emu_hmf.box.check({"H0": 55.0, "mnu": 0.4})
   ValueError: outside the CSST emulator's box, where this recalibration has no
   training data: H0 = 55 not in (60.0, 80.0); mnu = 0.4 not in (0.0, 0.3).
   The fit is not defined there and will not be extrapolated.

Curvature and the neutrino ordering
-----------------------------------

Two quantities reach a mass function without appearing in the box above.
``ggah_mod``'s :class:`Cosmology` carries :math:`\Omega_k`, which
:data:`emu_hmf.box.PARAMS` does not, so ``check_box`` cannot see a curved
cosmology and :func:`~emu_hmf.target.theta_from_cosmology` passes the eight
numbers it would pass for a flat one.  The box bounds :math:`\Sigma m_\nu`,
while ``nu_hierarchy`` fixes how that sum divides over three eigenstates; the
training set was generated with three equal masses.

Neither can be closed by retraining.  ``emu_pk`` closed the same gap against
CLASS, which solves a curved model on request, whereas this box belongs to a
simulation suite and would need curved simulations.

We measured the cost of both instead.  The measurement is possible because
:math:`\sigma(M)` is an input: curvature and the mass split reach a
multiplicity function through the variance and the growth history, which the
caller computes with the full parametrisation.  What remains is the response of
the correction itself, at fixed :math:`(\sigma, z)`.

.. note::

   These figures measure the response of the carrier, obtained by restoring the
   :math:`\Omega_k(1+z)^2` term the emulator's own :math:`E(z)` carries and
   re-evaluating its Castro+23 baseline at fixed :math:`\sigma`.  The CSST
   suite contains no curved simulations, so the response of the
   Gaussian-process residual is unmeasured.  We assume that curvature reaches a
   multiplicity function at fixed :math:`\sigma` through the growth history in
   :math:`\Omega_m(z)`, that the target's own model is parameterised in that
   quantity, and that no separate curvature dependence remains.  That is an
   assumption rather than a measurement.

:data:`emu_hmf.target.OMEGA_K_COST` holds the coefficient, as
:math:`\max|\Delta\ln f|` per unit :math:`|\Omega_k|`, and
:data:`~emu_hmf.target.NU_ORDERING_COST` the ordering shift.
``docs/make_validity_bounds.py`` regenerates both.

.. list-table::
   :header-rows: 1
   :widths: 16 16 22 16 16

   * - weights
     - coefficient
     - residual at 0.002
     - crossover
     - ``tinker08``
   * - ``200m``
     - 0.2242
     - 0.00520
     - 0.3009
     - 0.06766
   * - ``vir``
     - 0.8911
     - 0.00571
     - 0.1162
     - 0.10368

The crossover is the :math:`|\Omega_k|` at which the induced error, added in
quadrature to the held-out residual, reaches the ``tinker08`` this
recalibration replaces.  Below it the correction is the better answer; above it
the carrier is.  At the Planck and BAO bound of :math:`|\Omega_k| \le 0.002`
the 200m correction degrades from 0.00518 to 0.00520 against 0.06766 for
``tinker08``, a factor of thirteen, so this package passes curvature through
rather than refusing it.

The virial coefficient is 3.97 times the 200m one, and its crossover at 0.1162
lies inside a range a sampler might reach where 200m's 0.3009 does not.  A
single pooled coefficient would understate virial fourfold.

We measured the coefficient to :math:`|\Omega_k| = 0.45` on both files
(:data:`~emu_hmf.target.OMEGA_K_MEASURED_TO`).  It falls across that range, by
23 per cent at 200m and 15 at virial, in the same direction throughout, so
reading it as linear overestimates the cost and places the crossover early.

The sweep brackets both crossings.
:data:`~emu_hmf.target.OMEGA_K_MEASURED_CROSSING` holds each observed crossing
and :func:`~emu_hmf.target.crossover_is_measured` reports whether one exists.

.. list-table::
   :header-rows: 1
   :widths: 20 26 26 28

   * - weights
     - linear crossover
     - observed crossing
     - the law is early by
   * - ``200m``
     - 0.3009
     - 0.3715
     - 19 %
   * - ``vir``
     - 0.1162
     - 0.1210
     - 4 %

:mod:`emu_hmf.target` asserts at import that the linear crossover lies at or
below any observed crossing.  A threshold that erred late would recommend a
correction already worse than the carrier it replaces.

The ordering shift peaks at :math:`1.15\times10^{-4}` at the 0.058993 eV floor,
2.2 per cent of the residual, and falls to :math:`1.2\times10^{-5}` at the
ceiling of the box.  Added in quadrature it leaves the published 0.52 and 0.54
per cent unchanged at both figures they are quoted to.  A caller whose
:math:`\sigma(M)` came from a normal-ordered solve is closer to the physical
split than the training set is.

Only the caller knows the :math:`\Omega_k` and the ordering a cosmology was
built with, so this package records the cost rather than checking it.

The peak height: not checked, and narrower than it looks
--------------------------------------------------------

The fit was made over :math:`\nu = \delta_c/\sigma \in [0.5, 3]`
(:data:`emu_hmf.target.NU_TRUSTED`) and over
:math:`M \in [10^{12}, 10^{14}]\,M_\odot/h`
(:data:`emu_hmf.target.M_TRUSTED`).

The two cuts do not commute with redshift.  Growth pushes :math:`\sigma`
down, so a fixed mass is a higher peak later: the same
:math:`10^{12}\,M_\odot/h` that sits at :math:`\nu = 0.5` today sits at
:math:`\nu = 1.4` at :math:`z = 3`.  The low-:math:`\nu` half of the nominal
band is therefore simply *absent* from the training set above
:math:`z \simeq 0.25`.

.. figure:: _static/figures/covered_domain.png
   :width: 80%
   :align: center
   :alt: the covered peak-height band as a function of redshift

   Grey: inside the nominal range, and never sampled.  A caller who checked only
   :data:`~emu_hmf.target.NU_TRUSTED` would be extrapolating there with no
   warning.

:func:`emu_hmf.target.nu_covered` records what the training set actually spans:

.. code-block:: python

   >>> from emu_hmf import target
   >>> target.nu_covered(0.0)
   (0.5, 2.97)
   >>> target.nu_covered(3.0)
   (1.4, 3.0)

:math:`\sigma` reaches the package as a number the caller computed, from a
spectrum ``emu_hmf`` never sees, so this package records the range rather than
checking it.

Why the cut is in peak height
------------------------------

:math:`\nu` unifies mass and redshift.  The same :math:`\nu = 3` is
:math:`10^{15}\,M_\odot/h` at :math:`z = 0` and :math:`3\times10^{13}` at
:math:`z = 2`.  A cut in mass alone would keep the exponential tail at high
redshift, where a per-cent error in :math:`\sigma` is a tens-of-per-cent error
in :math:`f`, and discard signal at low redshift.

Why the upper mass limit is :math:`10^{14}`
--------------------------------------------

We measured the residual of a cubic in :math:`\ln M` through the target ratio
at the Planck fiducial:

.. list-table::
   :header-rows: 1
   :widths: 40 40

   * - upper limit [:math:`M_\odot/h`]
     - residual
   * - :math:`10^{14}`
     - 4.2e-3
   * - :math:`10^{14.5}`
     - 1.3e-2
   * - :math:`10^{15}`
     - 2.9e-2
   * - :math:`10^{15.5}`
     - 3.8e-2

The target is smooth to a few parts in a thousand up to
:math:`10^{14}\,M_\odot/h` and rougher above it, where the simulation suite
runs out of clusters and the Gaussian process has the thinnest training data.
The roughness belongs to the target rather than to any fit made to it.
``tests/test_target.py`` asserts both halves: smooth inside the range, rough
outside.

Redshift
--------

The emulator is trained at twelve redshifts from 0 to 3
(:data:`emu_hmf.target.Z_TRAINED`) and interpolates between them.  Above
:math:`z = 3` the correction is undefined.  Tinker08's own calibration stops at
:math:`z = 2.5`.
