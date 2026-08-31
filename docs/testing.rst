What the tests assert
=====================

.. code-block:: bash

   python -m pytest -q
   python -m pytest -q --cov=emu_hmf --cov-report=term-missing

The suite is written to pass in the environment ``pip install emu_hmf``
creates.  Tests that genuinely need the CSST emulator, CLASS or a halo-model
code skip rather than fail, so a full run in a plain install reports skips and
no failures.

Coverage, with the halo-model code available: **100 %**, every module.  In the
environment ``pip install emu_hmf`` creates it is 86 %, and the difference is
entirely the tests that skip there --- the ones that need a Gaussian-process
emulator or a Boltzmann solver to say anything.  Nothing is uncovered because
nobody wrote a test for it.

Test modules
------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - module
     - what it is for
   * - ``test_box.py``
     - the box is the emulator's own.  Copied rather than imported, and
       therefore checked against ``param_limits`` --- so the copy cannot drift
       without a test failing.
   * - ``test_model.py``
     - the inference path.  Tinker08 against the published table written out
       independently; :math:`g = 0` bit-for-bit; the gradient against a finite
       difference in all eight directions; ``jit`` and ``vmap``.
   * - ``test_target.py``
     - the dialect conversions, which are silently wrong if guessed, and the
       measured claim that the target is smooth inside
       :data:`~emu_hmf.target.M_TRUSTED` and rough outside it.
   * - ``test_fit.py``
     - the peak-height cut, the cosmology-wise split, the refusal to mix mass
       definitions, and an end-to-end fit that recovers a correction put there
       on purpose.
   * - ``test_generate.py``
     - the shard writer, with the solver stubbed: the atomic rename, the
       skip-if-exists resume, and that a refused cosmology is recorded rather
       than dropped.
   * - ``test_public_api.py``
     - the dependency split, in a subprocess.
   * - ``test_packaging.py``
     - builds a wheel and opens it.

Three assertions worth naming
------------------------------

**The split is load-bearing.**  ``test_public_api.py`` builds both corrections
in a fresh interpreter and fails if ``CEmulator``, ``classy``, ``optax``, scipy
or the halo-model code has appeared in ``sys.modules``.  The two-dependency
promise is checked, not merely stated.

**A held-out row is not a held-out cosmology.**  Each design contributes several
hundred rows --- twelve redshifts times the masses inside the peak-height cut
--- and at fixed cosmology :math:`\ln f` is a smooth function of :math:`\sigma`.
Split at random over rows and the network validates by interpolating between
neighbouring masses of a design it trained on.  That measures something real,
but it is not generalisation to a new cosmology, which is the only thing this
correction is ever asked for.  ``fit`` splits on cosmologies, the weights record
``split_by_cosmology``, and a test asserts it.

**A wheel without its weights installs and then raises.**  The trained networks
are package data, and a distribution that shipped the code without them would
pass every other test in this suite --- because every other test reads them out
of the working tree.  ``test_packaging.py`` builds the wheel, opens the archive,
installs it into an empty prefix and evaluates both corrections from a directory
that is not the repository.

Markers
-------

.. code-block:: bash

   python -m pytest -q -m "not slow"     # skip the wheel build and the training runs

``slow`` builds a wheel or trains a network; ``gen`` needs the training-set
stack.
