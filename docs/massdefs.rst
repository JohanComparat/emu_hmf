Two mass definitions, two files
===============================

There is no single correction with a :math:`\Delta` argument, and pretending
otherwise would put an error the size of the correction into whichever
definition was not fitted.

.. code-block:: python

   from emu_hmf.model import HmfCorrection, WEIGHTS

   m200 = HmfCorrection(WEIGHTS["200m"])     # the default
   vir  = HmfCorrection(WEIGHTS["vir"])

What each one is
----------------

.. list-table::
   :header-rows: 1
   :widths: 14 30 18 18 20

   * - key
     - halo definition
     - Tinker08 unchanged
     - recalibrated
     - improvement
   * - ``"200m"``
     - SO, 200 × mean, Rockstar
     - 7.00 %
     - **0.52 %**
     - 13.1 ×
   * - ``"vir"``
     - SO, virial, Rockstar
     - 10.92 %
     - **0.54 %**
     - 19.1 ×

rms in :math:`\ln f`, on 200 cosmologies held out entirely from training.  Both
numbers are read out of the weight files themselves --- ``val_rms`` and
``baseline_rms`` --- so they cannot drift from the weights they describe.

.. figure:: _static/figures/accuracy.png
   :width: 75%
   :align: center
   :alt: residual before and after recalibration, at both definitions

Both are *Rockstar* spherical-overdensity masses.  That is what makes comparing
the two corrections a test of the **boundary** and not of the halo finder ---
which is the whole reason the comparison is worth anything.

Why not 200c
------------

The emulator offers a third definition, ``FoFM200c``.  It is selectable in
:mod:`emu_hmf.target` and it is *not* what the shipped corrections were fitted
to, because it is a **friends-of-friends** mass.  Pairing a FoF mass with a
spherical-overdensity multiplicity function is a category error: the two count
different objects, not the same objects inside different radii.

So the fit is made at a true SO mass, at the definition Tinker08 was itself
calibrated in, and :math:`200{\rm c}` is reached afterwards through the
published :math:`\log\Delta` interpolation.

The virial weights are not a small correction
---------------------------------------------

This is the misreading worth guarding against.

Both files correct the **same carrier**: :func:`emu_hmf.model.tinker08`, which
is Tinker08 at :math:`\Delta_{\rm m} = 200`.  So the virial weights absorb the
change of boundary *as well as* the recalibration.  At :math:`z = 0` they sit
some 13 per cent below the 200m carrier, and the offset changes sign by
:math:`z \simeq 1` as :math:`\Delta_{\rm vir}(z)` falls toward the
Einstein--de Sitter value.

Read against Tinker08-at-200m, "correction" therefore means something different
in the two files:

* at ``"200m"`` it is a recalibration, and it is small;
* at ``"vir"`` it is a recalibration *plus* a definition change, and it is not.

Both reach the same residual against their own target --- around half a per
cent --- which is the number that says how well each does its job.

Do not mix them
---------------

A correction fitted at one definition, evaluated for halos defined at another,
is wrong by roughly the difference between the two definitions: larger than the
residual either achieves, and comparable to the offset they both correct.  The
two ``.npz`` files each record their own ``massdef``, and the training shards
do too, so a fit cannot silently average them:

.. code-block:: python

   >>> str(np.load(WEIGHTS["vir"])["massdef"])
   'RockstarMvir'
