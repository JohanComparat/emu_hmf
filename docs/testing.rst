What the tests assert
=====================

.. code-block:: bash

   python -m pytest -q
   python -m pytest -q --cov=emu_hmf --cov-report=term-missing

The suite passes in the environment ``pip install emu_hmf[dev]`` creates.  Tests
that need the CSST emulator, CLASS or a halo-model code skip rather than fail,
so a full run in a plain install reports skips and no failures.  Each such skip
states what stops being checked, because a skipped guard and a passing one read
alike in a summary line.

Coverage is 100 per cent of every module with the halo-model code available, and
87 per cent in the ``[dev]`` environment.  The difference is the tests that skip
there.

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

Four assertions
---------------

The dependency split.  ``test_public_api.py`` builds both corrections in a fresh
interpreter and fails if ``CEmulator``, ``classy``, ``optax``, ``emu_pk``,
``scipy``, ``matplotlib`` or the halo-model code appears in ``sys.modules``.

The validation split is over cosmologies, not rows.  Each design contributes
several hundred rows, twelve redshifts times the masses inside the peak-height
cut, and :math:`\ln f` is smooth in :math:`\sigma` at fixed cosmology.  A row
split therefore validates by interpolating in mass within a design the network
trained on.  ``fit`` splits on cosmologies, the weights record
``split_by_cosmology``, and a test asserts it.

The wheel carries its weights.  The trained networks are package data, and every
other test in this suite reads them out of the working tree.
``test_packaging.py`` builds the wheel, opens the archive, installs it into an
empty prefix and evaluates both corrections from outside the repository.

The measured constants match their archive.  ``test_target.py`` checks
:data:`~emu_hmf.target.OMEGA_K_COST` and its companions against
``docs/data/validity_bounds.npz`` field by field, and checks ``ggah_mod``'s
restatement of them against this package's copy where that package is
installed.

Markers
-------

.. code-block:: bash

   python -m pytest -q -m "not slow"     # skip the wheel build and the training runs

``slow`` builds a wheel or trains a network; ``gen`` needs the training-set
stack.
