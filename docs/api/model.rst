emu_hmf.model
=============

Inference: Tinker08 with cosmology-dependent parameters, in pure JAX.  This
module imports nothing beyond numpy and JAX, and is the only one a user of the
released package needs.

* :func:`~emu_hmf.model.tinker08` — the carrier, at
  :math:`\Delta_{\rm m} = 200`, optionally corrected.  With ``g=None`` it
  returns the published fit.
* :class:`~emu_hmf.model.HmfCorrection` — the recalibrated multiplicity
  function.  :meth:`~emu_hmf.model.HmfCorrection.fsigma` gives
  :math:`f(\sigma)`, :meth:`~emu_hmf.model.HmfCorrection.dndlnM` the abundance,
  and :meth:`~emu_hmf.model.HmfCorrection.g` the four log-corrections
  themselves.
* :data:`~emu_hmf.model.WEIGHTS` — one file per halo definition, keyed
  ``"200m"`` and ``"vir"``.  See :doc:`../massdefs`.
* :func:`~emu_hmf.model.load_weights` — the arrays and the provenance, cached
  and read-only.
* :func:`~emu_hmf.model.normalise` — :math:`(\theta, z)` onto the unit cube.

.. automodule:: emu_hmf.model
   :members:
   :show-inheritance:
