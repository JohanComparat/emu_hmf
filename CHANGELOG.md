# Changelog

## 1.0.0 — 2026-08-29

First public release.

* Recalibrated Tinker08 multiplicity function at two halo definitions,
  `RockstarM200m` and `RockstarMvir`, each fitted on 2000 CSST cosmologies over
  twelve redshifts and 456 526 rows inside ν ∈ [0.5, 3].
  Residual in ln f, on cosmologies held out entirely: 7.00 % → 0.52 % at 200m,
  10.92 % → 0.54 % at virial.
* Pure-JAX inference path: `jax.grad`, `jax.jit` and `jax.vmap` all pass through
  the cosmology. Two runtime dependencies, numpy and JAX.
* `target.nu_covered(z)` records the peak heights the training set actually
  spans, which narrows with redshift as growth pushes σ down.
* Both weight files are float64 and carry their own provenance — mass
  definition, sample size, held-out split and residual.
