Installation
============

.. code-block:: bash

   pip install emu_hmf

Using the package needs Python 3.10 or newer, numpy and JAX, and the 94 kB of
trained weights that ship inside the wheel, 47 kB per mass definition.

Do I need an environment file?
------------------------------

No.  ``import emu_hmf`` pulls in no compiler, conda channel, Boltzmann solver or
Gaussian-process emulator.  ``tests/test_public_api.py`` builds both corrections
in a fresh interpreter and fails if ``CEmulator``, ``classy``, ``optax``,
``emu_pk``, ``scipy``, ``matplotlib`` or the halo-model code appears in
``sys.modules``.

Extras
------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Extra
     - Adds
     - For
   * - *(none)*
     - numpy, JAX
     - evaluating the recalibrated mass function
   * - ``[train]``
     - optax
     - refitting the weights from an archived training set
   * - ``[dev]``
     - pytest, pytest-cov, optax, build (and ``tomli`` on Python 3.10)
     - running the test suite
   * - ``[docs]``
     - sphinx, sphinx-rtd-theme, myst-parser, matplotlib
     - building these pages and regenerating their figures

``[dev]`` carries ``optax`` because ``tests/test_fit.py`` refits a small network
end to end, so "running the test suite" is not satisfiable without it.  Note
``docs/requirements.txt``, which Read the Docs installs, omits matplotlib and
builds the pages against the committed figures.

.. code-block:: bash

   pip install "emu_hmf[train]"      # enough to reproduce the shipped weights
   pip install -e ".[dev]"           # a checkout, with the tests

Regenerating the training set, as opposed to refitting the network on it,
needs CLASS, the CSST emulator and the halo-model code.  Neither the emulator
nor the halo-model code is distributed on PyPI, so an extra for them could
never resolve, and the recipe lives in ``environment-gen.yml`` and in
:doc:`reproducing`.

From a checkout
---------------

.. code-block:: bash

   git clone https://github.com/JohanComparat/emu_hmf
   cd emu_hmf
   pip install -e ".[dev]"
   python -m pytest -q

Tests that need the generation stack skip rather than fail, so a full run in a
plain install reports skips and no failures.
