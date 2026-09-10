Use with a halo-model code
==========================

``emu_hmf`` stops at :math:`f(\sigma)`.  It takes a :math:`\sigma(M)` and
returns a multiplicity function, and carries no power spectrum, cosmology class
or halo model.

What the caller has to supply
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - quantity
     - convention that must match
   * - ``sigma``
     - the **cold** field (:math:`b + {\rm cdm}`, neutrinos excluded),
       integrated against :math:`\bar\rho_{cb}`
   * - ``dlnsigma_dlnm``
     - the logarithmic derivative of that same :math:`\sigma`, ideally by
       autodiff of the same integral rather than by differencing it
   * - ``rho_cold``
     - :math:`\bar\rho_{cb} = \Omega_{cb}\,\rho_{\rm crit,0}`, comoving,
       in :math:`(M_\odot/h)/({\rm Mpc}/h)^3`
   * - ``theta``
     - the eight CSST parameters, ``Omega_cb`` cold, amplitude as
       :math:`10^9 A_s`

A mismatch in any of them leaves every number plausible and the abundance
wrong by more than the correction being applied.

Reference: ``ggah_mod``
-----------------------

``ggah_mod`` is the halo-model code this package was built for.  It registers
the two corrections as named multiplicity functions:

.. code-block:: python

   make_field(..., hmf_model="tinker08_csst")        # the 200m weights
   make_field(..., hmf_model="tinker08_csst_vir")    # the virial weights

Four things it does that any integration should do.

It passes the cosmology through.  These two entries are registered as
cosmology-dependent, so the caller does not hand over ``theta`` explicitly;
every other entry in that registry is a function of :math:`(\sigma, z)` alone.

It converts the cosmology in one place.
:func:`emu_hmf.target.theta_from_cosmology` is the only translation from that
package's ``Cosmology`` into the eight parameters.  It calls no ``float()``,
because it runs inside a differentiable path.  Its inverse,
:func:`~emu_hmf.target.to_ggah_cosmology`, lives beside it, and
``tests/test_target.py`` pins the round trip in both directions.

The conversions require ``ggah_mod`` at :data:`emu_hmf.target.GGAH_MIN_VERSION`
or above, for :attr:`Omega_nu_matter` and the ``nu_hierarchy`` field.  They
probe the class for both rather than reading a version string.

It refuses to mix definitions.  Each registry entry declares the halo
definition it is calibrated for, and using ``tinker08_csst`` at ``mdef="vir"``
raises rather than returning an answer wrong by ten per cent.  See
:doc:`massdefs`.

It acts on the curvature cost.  ``ggah_mod`` reads
:data:`emu_hmf.target.OMEGA_K_COST` and its companions, and refuses a curved
cosmology past :data:`~emu_hmf.target.OMEGA_K_CROSSOVER`, where the correction
stops improving on the carrier it replaces.  See :doc:`validity`.

Rolling your own
----------------

.. code-block:: python

   from emu_hmf.model import HmfCorrection, WEIGHTS

   class MyHmf:
       def __init__(self, mdef="200m"):
           if mdef not in WEIGHTS:
               raise ValueError(f"emu_hmf has no weights for {mdef!r}")
           self.mdef = mdef
           self.corr = HmfCorrection(WEIGHTS[mdef])

       def dndlnM(self, m, sigma, dlns, rho_cold, theta, z):
           # your own guard here: nu inside target.nu_covered(z),
           # and m inside target.M_TRUSTED
           return self.corr.dndlnM(m, sigma, dlns, rho_cold, theta, z)

Keep ``check_box=True``.  It is skipped under tracing, so it costs nothing in
a compiled path.
