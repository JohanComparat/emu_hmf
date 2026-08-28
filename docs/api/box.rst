emu_hmf.box
===========

The hypercube the recalibration lives in, taken from the CSST emulator rather
than chosen, and copied here rather than imported so that evaluating a mass
function does not require installing a Gaussian-process emulator to find out
what the bounds are.  ``tests/test_box.py`` asserts the copy against the
emulator's own ``param_limits``.

* :data:`~emu_hmf.box.PARAMS` — the column order, which is the emulator's own.
* :data:`~emu_hmf.box.BOX` — the closed bounds.  Note that ``Omegam`` is the
  **cold** density.
* :func:`~emu_hmf.box.sample` — a Latin hypercube, deterministic in its seed.
* :func:`~emu_hmf.box.check` — raise, naming every parameter outside the box.
* :func:`~emu_hmf.box.inside` — the same, as data rather than an exception.

.. automodule:: emu_hmf.box
   :members:
   :show-inheritance:
