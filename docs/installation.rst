Installation
============

.. code-block:: bash

   pip install emu_hmf

That is the whole story for using the package.  Python 3.10 or newer, numpy and
JAX, and roughly 90 kB of trained weights that ship inside the wheel.

Do I need an environment file?
------------------------------

**No.**  There is no compiler, no conda channel, no Boltzmann solver and no
Gaussian-process emulator behind an ``import emu_hmf``.  The dependency list is
two entries long and that is deliberate: a forecast that wants to *evaluate* a
mass function should not be made to install the machinery that fitted one.

The split is asserted rather than assumed.  ``tests/test_public_api.py`` builds
both corrections in a fresh interpreter and fails if ``CEmulator``, ``classy``,
``optax`` or the halo-model code has appeared in ``sys.modules``.

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
     - pytest, pytest-cov, build
     - running the test suite
   * - ``[docs]``
     - sphinx, matplotlib
     - building these pages and regenerating their figures

.. code-block:: bash

   pip install "emu_hmf[train]"      # enough to reproduce the shipped weights
   pip install -e ".[dev]"           # a checkout, with the tests

Regenerating the *training set* --- as opposed to refitting the network on it
--- needs CLASS and the CSST emulator, and the emulator is not distributed on
PyPI.  Declaring it as an extra would produce one that could never resolve, so
the recipe is a file, ``environment-gen.yml``, and a page:
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
