What is being learned
=====================

Not a mass function.  A correction to Tinker08's four shape parameters.

The form is unchanged
---------------------

.. math::

   f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]
               e^{-c/\sigma^{2}}

with the published :math:`\Delta_{\rm m} = 200` values and redshift evolution,
and each of the four multiplied by :math:`e^{g_i(\theta, z)}`, where :math:`g`
is a small tanh network of the eight cosmological parameters and redshift.

Three things follow from writing it that way rather than fitting four free
functions of :math:`\sigma`:

* at :math:`g = 0` the answer **is** Tinker08, exactly.  So "the correction is
  zero" is a statement one can make and test, rather than a limit one hopes
  for; the test is in ``tests/test_model.py`` and it asserts bit-for-bit
  equality, not closeness.
* the peak-height dependence stays where the physics put it.  The network only
  has to express what the simulations add, which is a much smaller thing to
  learn than a mass function.
* the result is still a fit with named parameters.  A reader can ask what the
  recalibration did to the amplitude as against the tilt --- a question a black
  box cannot answer.

.. figure:: _static/figures/shape_parameters.png
   :width: 100%
   :align: center
   :alt: the four shape parameters as functions of redshift

   Where the recalibration goes.  Zero is the published fit.

Why the target is a ratio
-------------------------

The CSST emulator ships **both** its simulation-calibrated
:math:`\dd n/\dd\ln M` *and* its own Tinker08 evaluated against the cold
spectrum at an explicit :math:`\Delta`.  So the quantity to fit is their ratio,

.. math::

   R(M, z; \theta) = \frac{\dd n/\dd\ln M\ \big|_{\rm emulated}}
                          {\dd n/\dd\ln M\ \big|_{\rm Tinker08}}

which has two properties that a fit from scratch would not.  It is *small* --- a
few per cent at :math:`z = 0` --- so the fit is a correction rather than a
replacement, and the functional form keeps meaning what it meant.  And
:math:`R \to 1` at the calibration cosmology is a **testable** statement rather
than a hope.

The variance convention
-----------------------

:math:`\sigma(M)` is the **cold** field --- baryons and cold dark matter, with
massive neutrinos excluded --- integrated against :math:`\bar\rho_{cb}`, not
:math:`\bar\rho_m`.

This is not a detail.  A multiplicity function is only meaningful together with
the :math:`\sigma(M)` it was fitted against: change the convention and
:math:`f(\sigma)` is evaluated at a different :math:`\sigma` for the same halo,
which shifts the abundance by more than the correction being applied.  The
training set was therefore built by converting the emulator's abundance *into a
multiplicity function in this convention* at generation time, so the fit
absorbs the conversion once rather than leaving it to every caller.

If you supply a :math:`\sigma(M)` built from the total-matter spectrum, or
against :math:`\bar\rho_m`, the correction is being used outside the convention
it was fitted in and the package cannot tell.

The correction is not small at every redshift
---------------------------------------------

A few per cent at :math:`z = 0` is the number that prompted this package.  It is
not the number that describes it.

.. figure:: _static/figures/growth_with_redshift.png
   :width: 100%
   :align: center
   :alt: the correction grows with redshift

   Measured on the shipped 200m weights: the rms of :math:`\ln R` over the
   fitted band runs 2.4, 5.7, 9.5 and 11.7 per cent at
   :math:`z = 0, 1, 2, 3`.

Quoting the low-redshift figure alone would understate the correction
several-fold over most of the range it is defined on --- which is exactly why
:math:`g` is a function of :math:`z` and not a constant.

And it moves with the cosmology
-------------------------------

.. figure:: _static/figures/cosmology_dependence.png
   :width: 85%
   :align: center
   :alt: the correction for six cosmologies drawn from the box

   The premise of the package, as a picture.  A correction that did not depend
   on cosmology would be a constant --- and a constant is already inside
   Tinker08's amplitude.
