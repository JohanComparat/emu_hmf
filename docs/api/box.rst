emu_hmf.box
===========

The hypercube the recalibration is defined on, taken from the CSST emulator and
copied here rather than imported, so that evaluating a mass function needs no
Gaussian-process emulator.  ``tests/test_box.py`` asserts the copy against the
emulator's ``param_limits``.

* :data:`~emu_hmf.box.PARAMS` — the column order, taken from the emulator.
* :data:`~emu_hmf.box.BOX` — the closed bounds.  ``Omegam`` is the cold
  density.
* :func:`~emu_hmf.box.sample` — a Latin hypercube, deterministic in its seed.
* :func:`~emu_hmf.box.check` — raise, naming every parameter outside the box.
* :func:`~emu_hmf.box.inside` — the same, as data rather than an exception.

.. automodule:: emu_hmf.box
   :members:
   :show-inheritance:
