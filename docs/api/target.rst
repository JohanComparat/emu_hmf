emu_hmf.target
==============

What is being fitted, and the conversions that reach it.  The functions that
call the CSST emulator import it lazily, so this module is importable without
it — but calling them is not.

* :data:`~emu_hmf.target.FIDUCIAL` — a Planck-like point in the middle of the
  box; the cosmology every worked example and quoted number here uses.
* :data:`~emu_hmf.target.MASSDEFS`, :data:`~emu_hmf.target.DEFAULT_MASSDEF` —
  the three halo definitions the emulator offers, and what each one *is*.
* :data:`~emu_hmf.target.M_TRUSTED`, :data:`~emu_hmf.target.NU_TRUSTED`,
  :data:`~emu_hmf.target.DELTA_C` — the fitted domain.
* :data:`~emu_hmf.target.NU_COVERED`, :func:`~emu_hmf.target.nu_covered` — what
  the training set *actually* spans at each redshift, which is narrower.  See
  :doc:`../validity`.
* :func:`~emu_hmf.target.to_ggah_cosmology`,
  :func:`~emu_hmf.target.theta_from_cosmology` — the two directions of the
  cosmology conversion, and the only places it happens.
* :func:`~emu_hmf.target.csst_dndlnM`, :func:`~emu_hmf.target.csst_tinker08`,
  :func:`~emu_hmf.target.sigma_chain` — the offline half; these need the
  generation stack.

.. note::

   ``ggah_mod`` in the text below is the halo-model code this package was built
   for, and ``emu_pk`` its linear-spectrum emulator.  Neither is needed to
   *use* ``emu_hmf``; see :doc:`../halo_model`.

.. automodule:: emu_hmf.target
   :members:
   :show-inheritance:
