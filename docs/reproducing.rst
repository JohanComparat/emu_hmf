Reproducing the training set
============================

Two different jobs, with very different costs.

Refitting the weights: easy
---------------------------

Everything needed to turn the archived training set back into the shipped
weights is on PyPI:

.. code-block:: bash

   pip install "emu_hmf[train]"
   python -m emu_hmf.fit --shards ./shards --out weights.npz

A few minutes on a laptop CPU: 400 epochs over 456 526 rows of a
9 → 64 → 64 → 4 network.  The fit prints what it is doing and writes its own
provenance into the output:

.. code-block:: text

   8 shards of RockstarM200m, 2000 cosmologies (0 refused)
       -> 456526 rows in nu = [0.5, 3.0]
   held out 200 of 2000 cosmologies entirely (45666 of 456526 rows)
   Tinker08 unchanged, on the held-out split: rms 0.06766 in ln f (7.00%)
   ...
     Tinker08 unchanged : rms 0.06766 in ln f  (7.00%)
     recalibrated       : rms 0.00518 in ln f  (0.52%)
     improvement        : 13.07x

The fit pins ``jax_enable_x64`` itself, so the result does not depend on an
environment variable set somewhere else.  A network fitted in single precision
and one fitted in double are two different sets of weights, and the residual
being quoted is half a per cent.

The training data
-----------------

The shards are archived with a DOI --- 11.7 MB for both mass definitions, 16
files.  One shard holds 250 cosmologies:

.. list-table::
   :header-rows: 1
   :widths: 16 24 60

   * - key
     - shape
     - what it is
   * - ``idx``
     - ``(n_c,)``
     - index into the design, so a row can be traced back to its cosmology
   * - ``theta``
     - ``(n_c, 8)``
     - the cosmology, in :data:`emu_hmf.box.PARAMS` order
   * - ``f``
     - ``(n_c, 12, 24)``
     - the target: the emulator's abundance as a multiplicity function
   * - ``sigma``
     - ``(n_c, 12, 24)``
     - the cold-field :math:`\sigma`, against :math:`\bar\rho_{cb}`
   * - ``dlns``
     - ``(n_c, 12, 24)``
     - :math:`\dd\ln\sigma/\dd\ln M`
   * - ``z``
     - ``(12,)``
     - :data:`emu_hmf.target.Z_TRAINED`
   * - ``m``
     - ``(24,)``
     - :math:`M_\odot/h`, log-spaced over :data:`~emu_hmf.target.M_TRUSTED`
   * - ``failed_idx``
     - ``(n_f,)``
     - designs the solver refused.  A refusal is data, not a gap
   * - ``massdef``
     - scalar string
     - which halo definition these belong to

The design itself is not shipped and does not need to be: it is
``box.sample(2000, seed=20260828)``, deterministic in the seed, so a shard can
be rebuilt without the matrix and two shards can never disagree about which
index means which cosmology.

Regenerating the shards: expensive, and not pip-installable
------------------------------------------------------------

This is the half that needs a Boltzmann solver and the CSST emulator.  The
emulator is not distributed on PyPI, which is why there is no ``[gen]`` extra:
an extra that can never resolve is worse than a documented recipe.

.. code-block:: bash

   conda env create -f environment-gen.yml
   conda activate emu_hmf_gen

   python -m emu_hmf.generate --shard 0 --n-per-shard 250 --n-total 2000 \
          --out shards/hmf_000.npz --massdef RockstarM200m

Cost, measured on the shipped campaign: about 8000 s per 250 cosmologies, so
roughly 18 CPU-hours per mass definition and 35–40 for both.  Eight shards run
independently.  Shards write every ``CHUNK`` cosmologies and skip what is
already on disk, so a kill costs minutes rather than hours.

Why CLASS and not a spectrum emulator
--------------------------------------

The variance has to be available everywhere the CSST box goes.  A network
spectrum trained on :math:`\omega_b \in [0.017, 0.028]` covers only about 70 per
cent of a box that reaches :math:`\omega_b` from 0.0145 to 0.0382 --- and the
missing 30 per cent is not a corner but a slab.  CLASS has no box.  Generation
is offline and one-off, so paying a few seconds a cosmology to remove an
avoidable approximation from the training data is the easy side of that trade.
