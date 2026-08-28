emu_hmf.generate
================

Building the training set: the emulator's abundance, converted into a
multiplicity function in this package's variance convention.  Needs the
generation stack — see :doc:`../reproducing`.

* :func:`~emu_hmf.generate.solve_one` — one cosmology: one CLASS solve and one
  emulator call.
* :func:`~emu_hmf.generate.shard` — one contiguous slice of the design, written
  in chunks and resumable.
* :data:`~emu_hmf.generate.M_GRID`, :data:`~emu_hmf.generate.K_GRID`,
  :data:`~emu_hmf.generate.CHUNK` — the grids and the write cadence.

.. code-block:: bash

   python -m emu_hmf.generate --shard 0 --n-per-shard 250 --n-total 2000 \
          --out shards/hmf_000.npz --massdef RockstarM200m

.. automodule:: emu_hmf.generate
   :members:
   :show-inheritance:
