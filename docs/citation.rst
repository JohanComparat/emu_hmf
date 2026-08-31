Citation
========

If this package is useful, please cite three things: the functional form, the
simulations it is calibrated against, and the recalibration itself.

**The functional form.**  Tinker, J., Kravtsov, A. V., Klypin, A., et al. 2008,
*Toward a Halo Mass Function for Precision Cosmology: The Limits of
Universality*, ApJ 688, 709.  `doi:10.1086/591439 <https://doi.org/10.1086/591439>`_

**The target.**  Chen, Z. & Yu, Y. 2025, CSSTemu, the CSST emulator suite,
https://github.com/czymh/csstemu --- the emulated halo mass function this
package is recalibrated against, and the box it is defined on.  It installs
under its own name and imports as ``CEmulator``.

**This package.**  See ``CITATION.cff`` in the repository, which GitHub renders
as a *Cite this repository* button and which carries the archived DOI for the
release you used.

The training data is archived separately with its own DOI; cite it as well if
you refit rather than use the shipped weights.

What to say in a methods section
--------------------------------

Something along these lines, adjusted to what you actually used:

   Halo abundances were computed with the Tinker et al. (2008) multiplicity
   function, recalibrated against the CSST emulator (Chen & Yu 2025) using
   ``emu_hmf`` v1.0.0.  The recalibration multiplies each of the four Tinker08
   shape parameters by a cosmology- and redshift-dependent factor learned over
   the CSST parameter box, and reduces the residual against the emulated mass
   function from 7.0 to 0.5 per cent rms in :math:`\ln f` on held-out
   cosmologies.  Masses are spherical-overdensity :math:`M_{200{\rm m}}`, and
   :math:`\sigma(M)` is of the cold field against
   :math:`\bar\rho_{cb}`.

The last sentence is the one worth keeping: a multiplicity function is only
meaningful together with the variance convention it was fitted in.
