r"""A recalibrated Tinker08 multiplicity function, with cosmology dependence.

``tinker08`` is a fit to simulations, and not to the simulations anyone is
comparing against now: at the *Planck* cosmology it is offset, and the size of
the offset is itself a function of cosmology, which a fit with no cosmology
dependence beyond :math:`\sigma(M)` and :math:`z` cannot express.

What this package learns is therefore not a mass function.  It is a correction
to Tinker08's four shape parameters :math:`(A, a, b, c)`, as a function of the
eight cosmological parameters and redshift, trained against the CSST emulator
\-- ``CEmulator``, Chen & Yu (2025) -- over the box that emulator was built on.
Keeping Tinker08 as the carrier is the whole design:

* at zero correction the answer *is* Tinker08, exactly, so the baseline is a
  point in the same parameterisation rather than a different code;
* the peak-height dependence stays where the physics put it, and the network
  only has to express what the simulations add;
* it stays differentiable in the cosmology, which a table lookup would not.

The correction is defined only inside CSST's box and only over the peak heights
where the emulator itself is trustworthy (:data:`emu_hmf.target.NU_TRUSTED`).
Outside either, this package refuses rather than extrapolating -- the same
convention ``ggah_mod`` uses one rung down, and for the same reason.

Modules, in the order the work runs: :mod:`~emu_hmf.box` (the design space),
:mod:`~emu_hmf.target` (what is being fitted, and the dialect conversions that
reach it), :mod:`~emu_hmf.generate` (one CLASS solve and one emulator call per
design), :mod:`~emu_hmf.model` (Tinker08 plus the MLP), :mod:`~emu_hmf.fit`.
"""
__version__ = "0.2.0"

__all__ = ["box", "target", "model", "__version__"]
