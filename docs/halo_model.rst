Use with a halo-model code
==========================

``emu_hmf`` deliberately stops at :math:`f(\sigma)`.  It has no power spectrum,
no cosmology class and no halo model --- it takes a :math:`\sigma(M)` and
returns a multiplicity function.  Anything larger is the caller's.

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

A mismatch in any of them is silent: every number stays plausible and the
abundance is wrong by more than the correction being applied.

Reference: ``ggah_mod``
-----------------------

The halo-model code this package was built for is ``ggah_mod``, and it wires
the two corrections in as ordinary named multiplicity functions:

.. code-block:: python

   make_field(..., hmf_model="tinker08_csst")        # the 200m weights
   make_field(..., hmf_model="tinker08_csst_vir")    # the virial weights

Three things it does that any integration should do:

**It passes the cosmology through.**  These two entries are registered as
cosmology-dependent, so the caller does not have to remember to hand over
``theta``; everything else in that registry is a function of
:math:`(\sigma, z)` alone.

**It converts the cosmology in exactly one place.**
:func:`emu_hmf.target.theta_from_cosmology` is the only translation from that
package's ``Cosmology`` into the eight, and it is written to survive tracing ---
no ``float()`` anywhere in it --- because it is called from inside a
differentiable path.  Its inverse, :func:`~emu_hmf.target.to_ggah_cosmology`,
lives beside it and the round trip is pinned in both directions by
``tests/test_target.py``.

**It refuses to mix definitions.**  Each registry entry declares the halo
definition it is calibrated for, and a guard raises if the mass definition being
used does not match --- so ``tinker08_csst`` at ``mdef="vir"`` is a hard error
rather than a result that is wrong by ten per cent and looks fine.  Any
integration should carry the equivalent; see :doc:`massdefs`.

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

Keep ``check_box=True`` unless you are inside a ``jit`` that has already been
checked once --- it is skipped under tracing anyway, so it costs nothing in the
compiled path.
