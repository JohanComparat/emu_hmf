r"""A recalibrated Tinker08 multiplicity function, with cosmology dependence.

At the *Planck* cosmology ``tinker08`` is offset from the CSST suite, and the
offset varies with cosmology and redshift.  A fit whose only inputs are
:math:`\sigma(M)` and :math:`z` cannot express that variation.

This package fits a correction to Tinker08's four shape parameters
:math:`(A, a, b, c)` as a function of the eight cosmological parameters and
redshift, trained against the CSST emulator -- CSSTemu, Chen & Yu (2025) -- over
the box that emulator was built on.  Keeping Tinker08 as the carrier gives three
properties: at zero correction the result is Tinker08 exactly, the peak-height
dependence stays in the carrier, and the whole remains differentiable in the
cosmology.

The correction is defined inside CSST's box and over the peak heights the
training set reaches.  A cosmology outside the box raises, naming every
offending parameter.  The peak-height range cannot be checked here, because
:math:`\sigma` arrives as a number computed from a spectrum this package never
sees, so :func:`emu_hmf.target.nu_covered` records it instead.  That band
narrows with redshift: growth pushes :math:`\sigma` down, so the low-:math:`\nu`
end of :data:`~emu_hmf.target.NU_TRUSTED` is unsampled above
:math:`z \simeq 0.25`.

Curvature and the neutrino mass ordering have no axis in this box.
:data:`emu_hmf.target.OMEGA_K_COST` and
:data:`~emu_hmf.target.NU_ORDERING_COST` record what each costs.

Modules, in the order the work runs: :mod:`~emu_hmf.box` (the design space),
:mod:`~emu_hmf.target` (what is being fitted, and the conversions that reach
it), :mod:`~emu_hmf.generate` (one CLASS solve and one emulator call per
design), :mod:`~emu_hmf.model` (Tinker08 plus the network), :mod:`~emu_hmf.fit`.

Evaluating a mass function needs :mod:`~emu_hmf.model` and :mod:`~emu_hmf.box`
alone, which between them import nothing beyond numpy and JAX.
"""
__version__ = "1.2.0"

__all__ = ["box", "target", "model", "__version__"]
