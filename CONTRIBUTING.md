# Contributing

Issues and pull requests are welcome.

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The suite is meant to pass in an environment holding only the runtime
dependencies. Tests that need the generation stack — the CSST emulator, CLASS,
the halo-model code — skip rather than fail, so a full run in a plain install
reports skips and no failures.

```bash
python -m pytest -q --cov=emu_hmf --cov-report=term-missing
python -m pytest -q -m "not slow"      # skip the wheel build and the training runs
```

## What a change should come with

* A test that fails without it. Test names here are sentences, and docstrings
  say what would go wrong if the assertion did not hold — a number nobody checks
  becomes a number someone chose.
* Documentation, if the change is visible to a caller. `docs/` builds with
  `sphinx-build -W`, so a warning is an error.
* Two pytest markers exist: `slow`, and `gen` for tests that need the
  generation stack. Both are deselected with `-m "not slow and not gen"`.
* The measured constants in `emu_hmf.target` come out of
  `docs/make_validity_bounds.py`, which writes `docs/data/validity_bounds.npz`.
  Change either and regenerate the archive, then commit it: a test binds the
  constants to it, and the two disagreeing is the only way anyone finds out
  that a definition has moved. `--curvature-only` skips the CLASS half, which
  costs a Boltzmann solve per point and does not move when the curvature sweep
  is re-run.
* Any figure regenerated with `python docs/make_figures.py` and committed, so
  the documentation builds without the generation stack.

## Retraining

Changing the weights changes what everyone downstream computes. A pull request
that reships them should quote the held-out residual before and after, and say
which shards it was fitted on.
