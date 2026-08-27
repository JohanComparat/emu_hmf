# emu_hmf

A differentiable recalibration of the Tinker et al. (2008) multiplicity
function against the CSST halo-mass-function emulator, with cosmology
dependence.

`ggah_mod` reaches the CSST emulator today through `CsstHMF`, which is a
scikit-learn Gaussian process behind a numpy interface and is the halo layer's
*one* declared exception to differentiability.  This package exists to remove
that exception: same calibration, in a form a gradient can pass through.

## What is being learned, and why it is a ratio

The emulator ships **both** its simulation-calibrated `dn/dlnM` *and* its own
Tinker08 evaluated against the cold spectrum at an explicit Δ.  So the quantity
to fit is their ratio,

```
R(M, z; theta) = dn/dlnM |emulated  /  dn/dlnM |Tinker08
```

which has two properties a fit from scratch would not.  It is *small* — a few
per cent — so the fit is a correction rather than a replacement, and the
functional form keeps meaning what it meant.  And `R → 1` at the calibration
cosmology is a **testable** statement rather than a hope.

Measured at the Planck fiducial, over 10¹²–10¹⁴ M⊙/h:

```
R = 1.010 to 1.033
```

which is the "Tinker08 is a bit offset in Planck cosmology" that prompted this
package, quantified.  And it moves with the cosmology, which is the whole
premise:

| cosmology | R over 10¹²–10¹⁵ |
|---|---|
| Planck-ish | 0.978 – 1.034 |
| Ω_m = 0.25 | 0.932 – 1.067 |
| w = −0.8 | 0.988 – 1.071 |
| Σm_ν = 0.3 | 0.942 – 1.030 |

## Three decisions taken before any fitting

**The box is CSST's.**  Ω_b [0.04, 0.06], Ω_m [0.24, 0.40], H₀ [60, 80],
n_s [0.92, 1.00], 10⁹A_s [1.7, 2.5], w [−1.3, −0.7], w_a [−0.5, 0.5],
Σm_ν [0, 0.3].  Copied into `box.py` rather than imported, because a package
meant to be importable by a forecast cannot make the forecast install a
Gaussian-process emulator to find out its own bounds — and `tests/test_box.py`
asserts the copy against `CEmulator`'s own `param_limits`, so it cannot drift.

It is **narrower than `emu_pk`'s** in three axes, which matters because the
σ(M) this recalibrates against comes from there: `emu_pk` reaches h = 0.55
where CSST starts at H₀ = 60, and Σm_ν = 0.6 where CSST stops at 0.3.

**The mass definition is `RockstarM200m`, not `FoFM200c`.**  The feedback that
prompted this package asks for 200c — but the only 200c the emulator has is a
*friends-of-friends* mass, and pairing a FoF mass with a spherical-overdensity
multiplicity function is the category error `ggah_mod.halos.calibration`
refuses one rung down.  So the fit is made at a true SO mass, at the definition
Tinker08 was itself calibrated in, and 200c is reached afterwards through the
published log Δ interpolation exactly as `ggah_mod` already does it.
`FoFM200c` stays selectable and documented as carrying a finder change along
with the definition.

**The variance is ours, not the emulator's.**  σ(M) comes from `ggah_mod` on
`emu_pk`'s spectrum — the cold field against ρ̄_cb — because that is the σ(M)
the recalibrated fit will be *evaluated* with.  Fitting f(σ) against one
variance and using it with another is the mismatch that makes a multiplicity
function look wrong when the convention around it is what moved.

## Where the target can be trusted, measured

The residual of a cubic in ln M through R at the Planck fiducial:

| upper limit | residual |
|---|---|
| 10¹⁴ | 4.2e-3 |
| 10¹⁴·⁵ | 1.4e-2 |
| 10¹⁵ | 3.0e-2 |
| 10¹⁵·⁵ | 4.0e-2 |

So the target is smooth to a few parts in a thousand up to 10¹⁴ M⊙/h and
progressively rougher above it, reaching R = 0.91 by 10¹⁵·⁵.  That is where a
simulation suite runs out of clusters and where a Gaussian process is noisiest,
so it is a property of the **target** and not of any fit made to it.  A
recalibration claimed to a per cent above that range would be claiming to
reproduce the emulator's own noise.  `target.M_TRUSTED` is that range, and
`tests/test_target.py` checks both halves of the statement — smooth inside,
rough outside — because a range nobody checks becomes a number someone chose.

## Status

Scaffold: the box, the target, and the conversions between the three packages'
conventions, all under test.  The fit itself is next.
