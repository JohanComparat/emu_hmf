# emu_hmf

A differentiable, cosmology-dependent recalibration of the Tinker et al. (2008)
halo multiplicity function, trained against the CSST emulator
([Chen & Yu 2025](https://github.com/czymh/CEmulator)) over the box that
emulator was built on.

Tinker08 is a fit to simulations — and not to the simulations anyone compares
against now. At a Planck cosmology it is offset by a few per cent at z = 0, and
the size of that offset is itself a function of cosmology and redshift, which a
fit whose only inputs are σ(M) and z cannot express. This package learns that
offset.

What it learns is *not* a mass function. It is a correction to Tinker08's four
shape parameters (A, a, b, c) as a function of the eight CSST cosmological
parameters and redshift:

```
f(σ) = A [ (σ/b)^-a + 1 ] exp(-c/σ²),   with   (A, a, b, c) → (A, a, b, c) · e^g(θ, z)
```

Keeping Tinker08 as the carrier is the whole design. At `g = 0` the answer *is*
Tinker08, exactly, so the baseline is a point in the same parameterisation
rather than a different code. The peak-height dependence stays where the physics
put it, and the network only has to express what the simulations add. And the
result is a fit with named parameters, so you can ask what the recalibration did
to the amplitude as against the tilt.

## Install

```bash
pip install emu_hmf
```

Two dependencies, numpy and JAX, and 90 kB of trained weights. No Boltzmann
solver, no Gaussian-process emulator, no training stack, no conda environment —
a forecast that wants to *evaluate* a mass function should not have to install
the machinery that fitted one. `tests/test_public_api.py` asserts that split
rather than trusting it.

## Use

```python
import numpy as np
from emu_hmf.model import HmfCorrection

corr = HmfCorrection()                       # 200m; HmfCorrection(WEIGHTS["vir"]) for virial

theta = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.06])
#                 Ω_b    Ω_cb  H0     n_s     10⁹A_s  w    w_a  Σm_ν

f = corr.fsigma(sigma=0.8, theta=theta, z=0.5)          # the multiplicity function
n = corr.dndlnM(m, sigma, dlnsigma_dlnm, rho_cold, theta, z=0.5)   # the abundance
```

σ(M) is passed in, not computed: this package has no power spectrum and should
not acquire one, and the σ(M) the fit was made against is the *cold* field
against ρ̄_cb. Fitting f(σ) against one variance and evaluating it with another
is the mismatch that makes a multiplicity function look wrong when the
convention around it is what moved.

Everything is JAX, so `jax.grad`, `jax.jit` and `jax.vmap` all work through the
cosmology. That is the reason this exists rather than a table of numbers.

## Two mass definitions, two files

The correction is not the same function at two halo boundaries, so there is no
single correction with a Δ argument. Both are fitted against a *Rockstar*
spherical-overdensity mass, so the comparison isolates the boundary rather than
mixing in a change of halo finder.

| weights | halo definition | Tinker08 unchanged | recalibrated | improvement |
| --- | --- | --- | --- | --- |
| `WEIGHTS["200m"]` | SO 200 × mean, Rockstar | 7.00 % | **0.52 %** | 13.1× |
| `WEIGHTS["vir"]` | SO virial, Rockstar | 10.92 % | **0.54 %** | 19.1× |

rms in ln f, on 200 cosmologies held out *entirely* from training — not held-out
rows. Each design contributes several hundred rows and at fixed cosmology ln f
is smooth in σ, so a random row split measures interpolation between neighbouring
masses of a cosmology the network has already seen. This correction is only ever
asked for a cosmology it has not seen.

Both files carry `WEIGHTS["200m"]`'s Δ = 200m Tinker08 as the carrier, so the
virial weights absorb the *boundary change* as well as the recalibration. They
are not a per-cent correction: at z = 0 they sit some 13 % below the 200m
carrier. Reading "correction" as "small" at `vir` is a misreading.

## Where it is defined

Outside either bound the package refuses rather than extrapolating.

**The cosmology** must be inside CSST's box, which is copied into `box.py` and
checked against the emulator's own `param_limits` by `tests/test_box.py`:

| Ω_b | Ω_cb | H₀ | n_s | 10⁹A_s | w | w_a | Σm_ν |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.04–0.06 | 0.24–0.40 | 60–80 | 0.92–1.00 | 1.7–2.5 | −1.3–−0.7 | −0.5–0.5 | 0–0.3 |

Note Ω_cb: CSST bounds the **cold** density, with massive neutrinos excluded.

**The peak height** must be inside ν = δ_c/σ ∈ [0.5, 3], and the mass inside
10¹²–10¹⁴ M⊙/h. Those two cuts do not commute with redshift: growth pushes σ
down, so a fixed mass is a higher peak later, and the low-ν half of the band is
simply absent above z ≈ 0.25. `target.nu_covered(z)` records what the training
set actually spans — ν ≥ 1.4 by z = 3 — because a caller who checked only the
nominal range would be extrapolating with no warning.

The correction is a few per cent at z = 0 and grows with redshift, reaching
about 12 % rms by z = 3. Quoting the low-redshift figure alone would understate
it several-fold over most of the range it is defined on.

## Documentation

[emu-hmf.readthedocs.io](https://emu-hmf.readthedocs.io) — concepts, a tutorial
with figures, the validity domain, and how to reproduce the training set.

## Reproducing

The 2000-cosmology training set (11.7 MB, both mass definitions) is archived
with a DOI; the fit that turns it into the shipped weights needs only `optax`:

```bash
pip install emu_hmf[train]
python -m emu_hmf.fit --shards ./shards --out weights.npz
```

Regenerating the shards themselves needs CLASS and the CSST emulator; see the
documentation's *Reproducing the training set* page.

## Citation

If you use this package, please cite Tinker et al. (2008) for the functional
form, Chen & Yu (2025) for the CSST emulator this is calibrated against, and
this package for the recalibration. See `CITATION.cff`.

## Licence

MIT. See [LICENSE](LICENSE).
