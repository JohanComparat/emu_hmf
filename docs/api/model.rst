emu_hmf.model
=============

Inference: Tinker08 with cosmology-dependent parameters, in pure JAX.  This is
the only module a user of the released package needs, and the only one that
imports nothing beyond numpy and JAX.

* :func:`~emu_hmf.model.tinker08` — the carrier, at
  :math:`\Delta_{\rm m} = 200`, optionally corrected.  With ``g=None`` it is the
  published fit, unchanged.
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
