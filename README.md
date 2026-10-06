# emu_hmf

[![tests](https://github.com/JohanComparat/emu_hmf/actions/workflows/ci.yml/badge.svg)](https://github.com/JohanComparat/emu_hmf/actions/workflows/ci.yml)
[![docs](https://readthedocs.org/projects/emu-hmf/badge/?version=latest)](https://emu-hmf.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/emu-hmf?logo=pypi&logoColor=white)](https://pypi.org/project/emu-hmf/)
[![Python](https://img.shields.io/pypi/pyversions/emu-hmf?logo=python&logoColor=white)](https://pypi.org/project/emu-hmf/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://github.com/JohanComparat/emu_hmf/blob/main/LICENSE)

A differentiable, cosmology-dependent recalibration of the Tinker et al. (2008)
halo multiplicity function, trained against the CSST emulator
([Chen & Yu 2025](https://github.com/czymh/csstemu)) over the box that emulator
was built on.

At a Planck cosmology Tinker08 is offset from the CSST suite by a few per cent
at z = 0, and the offset varies with cosmology and redshift. A fit whose only
inputs are σ(M) and z cannot express that variation. We fit a correction to
Tinker08's four shape parameters as a function of the eight CSST cosmological
parameters and redshift:

```
f(σ) = A [ (σ/b)^-a + 1 ] exp(-c/σ²),   with   (A, a, b, c) → (A, a, b, c) · e^g(θ, z)
```

At `g = 0` this returns Tinker08 exactly. Keeping Tinker08 as the carrier
expresses the correction in its named parameters, which separates a change in
amplitude from a change in tilt.

## Install

```bash
pip install emu_hmf
```

The inference path needs numpy and JAX, and 94 kB of trained weights, 47 kB per
mass definition. It needs no Boltzmann solver, no Gaussian-process emulator and
no training stack; `tests/test_public_api.py` asserts the split.

### A dedicated environment

> **Maintainer setup.** On the development laptop, use the shared `dev` environment defined in `~/software/dev_env` (`conda activate dev`); this package is already installed there in editable mode. Do not create a separate environment for it: add missing dependencies to `~/software/dev_env` and rebuild.

[`environment.yml`](https://github.com/JohanComparat/emu_hmf/blob/main/environment.yml)
builds a reproducible one:

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
mamba activate emu_hmf
pip install -e .
```

It pins the CPU build of `jaxlib`, 64 MB against 199 MB for the CUDA one.
Unpinned, the build resolves against whichever driver the solving machine has.
The file says how to swap it for a GPU, and carries commented blocks for the
`[dev]`, `[train]` and `[docs]` extras.

## Use

```python
import numpy as np
from emu_hmf.model import HmfCorrection, WEIGHTS

corr = HmfCorrection()                       # 200m; HmfCorrection(WEIGHTS["vir"]) for virial

theta = np.array([0.049, 0.31, 67.36, 0.9649, 2.1, -1.0, 0.0, 0.06])
#                 Ω_b    Ω_cb  H0     n_s     10⁹A_s  w    w_a  Σm_ν

f = corr.fsigma(sigma=0.8, theta=theta, z=0.5)
```

`dndlnM(m, sigma, dlnsigma_dlnm, rho_cold, theta, z)` returns the abundance from
the same quantities.

The caller supplies σ(M); this package computes no power spectrum. The variance
the fit was made against is the cold field against ρ̄_cb, and evaluating f(σ)
against a different variance shifts the answer by the difference between the two
conventions.

`jax.grad`, `jax.jit` and `jax.vmap` all pass through the cosmology.

## Two mass definitions, two files

We fit one correction per halo definition. Both are fitted against a Rockstar
spherical-overdensity mass, so the comparison between them isolates the boundary
rather than the halo finder.

| weights | halo definition | Tinker08 unchanged | recalibrated | improvement |
| --- | --- | --- | --- | --- |
| `WEIGHTS["200m"]` | SO 200 × mean, Rockstar | 7.00 % | 0.52 % | 13.1× |
| `WEIGHTS["vir"]` | SO virial, Rockstar | 10.92 % | 0.54 % | 19.1× |

The figures are rms in ln f, measured on 200 cosmologies held out entirely from
training. We split on cosmologies rather than rows because each design
contributes several hundred rows and ln f is smooth in σ at fixed cosmology, so
a row split measures interpolation in mass.

Both files carry the Δ = 200m Tinker08 as the carrier, so the virial weights
absorb the change of boundary as well as the recalibration. At z = 0 they sit
14 % below that carrier on average across the covered peak-height band, ranging
from 8 % to 24 % with peak height.

## Where it is defined

**The cosmology** must lie inside CSST's box, which is copied into `box.py` and
checked against the emulator's own `param_limits` by `tests/test_box.py`:

| Ω_b | Ω_cb | H₀ | n_s | 10⁹A_s | w | w_a | Σm_ν |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.04–0.06 | 0.24–0.40 | 60–80 | 0.92–1.00 | 1.7–2.5 | −1.3–−0.7 | −0.5–0.5 | 0–0.3 |

Ω_cb is the cold density, with massive neutrinos excluded.

**The peak height** must lie inside ν = δ_c/σ ∈ [0.5, 3], and the mass inside
10¹²–10¹⁴ M⊙/h. The two cuts do not commute with redshift: growth pushes σ
down, so a fixed mass is a higher peak later, and the low-ν half of the band is
absent above z ≈ 0.25. `target.nu_covered(z)` records what the training set
spans, reaching ν ≥ 1.4 by z = 3.

The package refuses a cosmology outside the box.  It cannot check the peak
height, because σ arrives as a number the caller computed, so it records the
covered range instead.

**Curvature and the neutrino mass ordering** have no axis in this box, because
the CSST suite is flat and carries one neutrino species. We measured what each
costs and pass both through.

| weights | cost per unit \|Ω_k\| | crossover | observed crossing | at \|Ω_k\| = 0.002 |
| --- | --- | --- | --- | --- |
| `200m` | 0.2242 | 0.3009 | 0.3715 | 0.00520 vs 0.06766 |
| `vir` | 0.8911 | 0.1162 | 0.1210 | 0.00571 vs 0.10368 |

The crossover is where the induced error, added in quadrature to the held-out
residual, reaches the Tinker08 this replaces. It is computed from the
coefficient read as linear, which places it below the crossing the sweep
observes. `ggah_mod` refuses past the crossover.

At the Planck and BAO bound of \|Ω_k\| ≤ 0.002, refusing a curved cosmology
costs a factor of thirteen against accepting one. The virial coefficient is
3.97 times the 200m one, so the two do not pool. The ordering leaves the
correction unchanged, because the matter budget is invariant under the split,
and moves the target by at most 2.2 % of the residual.

These figures measure the response of the Castro+23 carrier the emulator is
built on. The CSST suite contains no curved simulations, so the response of the
Gaussian-process residual is unmeasured. `target.OMEGA_K_COST` and its
companions carry the numbers, and the [validity
page](https://emu-hmf.readthedocs.io/en/latest/validity.html) states the limits.

The two files behave differently with redshift.  The 200m correction grows
from 2.3 % rms at z = 0 to 11.6 % at z = 3.  The virial correction is U-shaped,
16.4 % at z = 0, falling to 3.1 % near z = 0.7 and returning to 16.3 % by
z = 3, because it carries the change of boundary as well as the
recalibration.

## Documentation

[emu-hmf.readthedocs.io](https://emu-hmf.readthedocs.io) — concepts, a tutorial
with figures, the validity domain, and how to reproduce the training set.

## Reproducing

The 2000-cosmology training set is 11.7 MB for both mass definitions. Fitting it
into the shipped weights needs `optax`:

```bash
pip install "emu_hmf[train]"
python -m emu_hmf.fit --shards ./shards --out weights.npz
```

Regenerating the shards needs CLASS and the CSST emulator; see the
documentation's *Reproducing the training set* page.

## Citation

Please cite Tinker et al. (2008) for the functional form, Chen & Yu (2025) for
the CSST emulator, and this package for the recalibration. See `CITATION.cff`.

## Licence

BSD 3-Clause. See [LICENSE](https://github.com/JohanComparat/emu_hmf/blob/main/LICENSE).
