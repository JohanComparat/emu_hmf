Reproducing the training set
============================

Refitting the weights and regenerating the shards are separate jobs with very
different costs.

Refitting the weights
---------------------

Turning the archived training set back into the shipped weights needs only what
is on PyPI:

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

The fit pins ``jax_enable_x64`` itself, so the weights do not depend on an
environment variable set elsewhere.  Single and double precision give different
weights, against a quoted residual of half a per cent.

The training data
-----------------

The shards are 11.7 MB for both mass definitions, 16 files.  One shard holds 250 cosmologies:

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
emulator is not distributed on PyPI, so there is no ``[gen]`` extra:
an extra that can never resolve is worse than a documented recipe.

.. code-block:: bash

   conda env create -f environment-gen.yml
   conda activate emu_hmf_gen

   python -m emu_hmf.generate --shard 0 --n-per-shard 250 --n-total 2000 \
          --out shards/hmf_000.npz --massdef RockstarM200m

``environment-gen.yml`` pins ``ggah_mod`` to a tag.  Two of its conventions
determine what a generated shard contains:

* :attr:`Omega_cb` --- which sets :math:`\bar\rho_{cb}`, and so
  :math:`\sigma(M)` --- subtracts the matter-like part of the Fermi-Dirac
  neutrino density.  It used to subtract the 93.14 eV convention, which sits
  :math:`4.6\times10^{-3}` below it.
* ``nu_hierarchy`` fixes how :math:`\Sigma m_\nu` divides over three
  eigenstates.  :func:`~emu_hmf.target.to_ggah_cosmology` pins it to
  ``"degenerate"``, three equal masses, the convention the shipped weights were
  fitted under.

Both landed after ``ggah_mod`` last tagged, so no version comparison
distinguishes a tree that carries them.  The conversions probe the class instead
and refuse a release too old for either.
:data:`emu_hmf.target.GGAH_MIN_VERSION` names the release for the error
message.

Cost, measured on the shipped campaign: about 8000 s per 250 cosmologies, so
roughly 18 CPU-hours per mass definition and 35–40 for both.  Eight shards run
independently.  Shards write every ``CHUNK`` cosmologies and skip what is
already on disk, so a kill costs minutes rather than hours.

Why CLASS and not a spectrum emulator
--------------------------------------

The variance must be available everywhere the CSST box reaches.  ``emu_pk`` is
trained on :math:`\omega_b \in [0.017, 0.028]` and this box spans 0.0147 to
0.0382, so 30 per cent of the design falls outside it, above and below.  CLASS
carries no such bound.  Generation is offline and costs about 32 s a cosmology,
which buys the training set an exact spectrum.  ``tests/test_box.py`` measures the
coverage.
