---
name: Bug report
about: Something behaves differently from what the documentation says
labels: bug
---

**What happened, and what you expected instead**

**A minimal example**

```python
```

**Where you were evaluating it**

The cosmology (the eight parameters), the redshift, and the peak height
`nu = 1.686 / sigma`. Please check it against `target.nu_covered(z)` — above
z ≈ 0.25 the low-nu half of the nominal band is not in the training set, and
the package cannot detect that for you.

**Which weights** — `"200m"` or `"vir"`.

**Versions** — `emu_hmf`, `jax`, `numpy`, Python, OS.
