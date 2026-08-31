r"""A recalibrated Tinker08 multiplicity function, with cosmology dependence.

``tinker08`` is a fit to simulations, and not to the simulations anyone is
comparing against now: at the *Planck* cosmology it is offset, and the size of
the offset is itself a function of cosmology, which a fit with no cosmology
dependence beyond :math:`\sigma(M)` and :math:`z` cannot express.

What this package learns is therefore not a mass function.  It is a correction
to Tinker08's four shape parameters :math:`(A, a, b, c)`, as a function of the
eight cosmological parameters and redshift, trained against the CSST emulator
\-- CSSTemu, Chen & Yu (2025) -- over the box that emulator was built on.
Keeping Tinker08 as the carrier is the whole design:

* at zero correction the answer *is* Tinker08, exactly, so the baseline is a
  point in the same parameterisation rather than a different code;
* the peak-height dependence stays where the physics put it, and the network
  only has to express what the simulations add;
* it stays differentiable in the cosmology, which a table lookup would not.

The correction is defined only inside CSST's box and only over the peak heights
the training set reaches.  A cosmology outside the box raises, naming every
offending parameter; the peak-height half cannot be checked here, because
:math:`\sigma` arrives as a number computed from a spectrum this package never
sees, so :func:`emu_hmf.target.nu_covered` records it instead.  Note that the
covered band narrows with redshift: growth pushes :math:`\sigma` down, so the
low-:math:`\nu` end of :data:`~emu_hmf.target.NU_TRUSTED` is not sampled above
:math:`z \simeq 0.25`.

Modules, in the order the work runs: :mod:`~emu_hmf.box` (the design space),
:mod:`~emu_hmf.target` (what is being fitted, and the conversions that reach
it), :mod:`~emu_hmf.generate` (one CLASS solve and one emulator call per
design), :mod:`~emu_hmf.model` (Tinker08 plus the network), :mod:`~emu_hmf.fit`.

Only :mod:`~emu_hmf.model` and :mod:`~emu_hmf.box` are needed to *evaluate* a
mass function, and between them they import nothing beyond numpy and JAX.  The
other two are the offline half that built the weights.
"""
__version__ = "1.0.0"

__all__ = ["box", "target", "model", "__version__"]
