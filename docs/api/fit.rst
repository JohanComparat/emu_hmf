emu_hmf.fit
===========

Fitting the four Tinker08 parameters as functions of cosmology and redshift.
Needs ``optax``, the ``[train]`` extra, and nothing further, so this step runs
from the archived shards alone.

* :class:`~emu_hmf.fit.Shards` — one flat training set, with the record of what
  it was built from.
* :func:`~emu_hmf.fit.load_shards` — every shard in a directory, cut to the
  trusted peak heights, refusing to mix mass definitions.
* :func:`~emu_hmf.fit.fit` — the optimisation.  Splits on whole cosmologies,
  pins float64, and writes its provenance into the output.

.. code-block:: bash

   python -m emu_hmf.fit --shards ./shards --out weights.npz

.. automodule:: emu_hmf.fit
   :members:
   :show-inheritance:
