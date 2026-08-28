r"""What a released install must be: two dependencies, and nothing hidden.

The dependency split is not a tidiness preference, it is the reason this
package exists separately from the code that trained it.  A forecast that wants
to *evaluate* a mass function must not be made to install a Boltzmann solver, a
Gaussian-process emulator or an optimiser -- so ``import emu_hmf`` in an
environment holding numpy and JAX alone has to work, and these are the
assertions that keep it true.
"""
import importlib.metadata
import importlib.resources
import pathlib
import subprocess
import sys
import tomllib

import numpy as np
import pytest

import emu_hmf
from emu_hmf import box, model, target

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: What evaluating the correction must not drag in.
FORBIDDEN = ("CEmulator", "ggah_mod", "emu_pk", "classy", "optax",
             "scipy", "matplotlib")


def _in_subprocess(source, **env):
    """Run ``source`` in a fresh interpreter and return its stdout."""
    import os

    e = dict(os.environ, PYTHONPATH=str(ROOT), **env)
    r = subprocess.run([sys.executable, "-c", source], capture_output=True,
                       text=True, env=e)
    assert r.returncode == 0, r.stderr
    return r.stdout


class TestTheSplitIsLoadBearing:
    def test_evaluating_the_correction_imports_nothing_it_should_not(self):
        """Checked in a subprocess, because by the time this test module has
        been collected the rest of the suite has already imported half of
        them."""
        out = _in_subprocess(f"""
import sys
import numpy as np
from emu_hmf.model import HmfCorrection, WEIGHTS
import emu_hmf.target                       # even the offline half is safe
for key in WEIGHTS:
    c = HmfCorrection(WEIGHTS[key])
    v = float(c.fsigma(1.0, np.array({list(target.FIDUCIAL)!r}), 0.5))
    assert np.isfinite(v), key
leaked = [m for m in {FORBIDDEN!r} if m in sys.modules]
print("LEAKED:" + ",".join(leaked))
""")
        leaked = out.strip().removeprefix("LEAKED:").strip()
        assert not leaked, (
            f"evaluating the correction imported {leaked}; the released "
            "package must need numpy and JAX and nothing else")

    def test_the_declared_dependencies_are_the_two(self):
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
        names = [d.split(">")[0].split("=")[0].split("[")[0].strip()
                 for d in meta["project"]["dependencies"]]
        assert sorted(names) == ["jax", "numpy"], names

    def test_the_generation_stack_is_not_a_declared_extra(self):
        """It cannot resolve: the CSST emulator is not distributed on PyPI.
        An extra that always fails is worse than a documented recipe."""
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for extra, deps in meta["project"]["optional-dependencies"].items():
            for d in deps:
                assert "CEmulator" not in d and "ggah_mod" not in d, (extra, d)


class TestTheVersion:
    def test_the_module_and_the_metadata_agree(self):
        """Two places state the version; a release that bumped one of them is
        a release whose weights and whose changelog disagree."""
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
        assert emu_hmf.__version__ == meta["project"]["version"]

    def test_the_installed_distribution_agrees_too(self):
        try:
            installed = importlib.metadata.version("emu_hmf")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("running from a source tree, not an install")
        assert installed == emu_hmf.__version__

    def test_it_is_a_release_version(self):
        assert emu_hmf.__version__.count(".") == 2
        assert all(p.isdigit() for p in emu_hmf.__version__.split("."))


class TestThePublicNames:
    @pytest.mark.parametrize("mod", [box, model, target])
    def test_everything_promised_exists(self, mod):
        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        assert not missing, f"{mod.__name__}.__all__ promises {missing}"

    def test_the_package_lists_its_modules(self):
        for name in emu_hmf.__all__:
            if name == "__version__":
                continue
            __import__(f"emu_hmf.{name}")


class TestTheWeightsAreFoundLikeAnInstalledPackage:
    def test_they_resolve_through_importlib_resources(self):
        """``model`` locates them by ``__file__``, which is right for a normal
        wheel; this asserts the same files are reachable the way an installed
        distribution is meant to be read."""
        data = importlib.resources.files("emu_hmf") / "data"
        for path in model.WEIGHTS.values():
            assert (data / path.name).is_file(), path.name

    def test_they_are_inside_the_package_directory(self):
        pkg = pathlib.Path(emu_hmf.__file__).resolve().parent
        for path in model.WEIGHTS.values():
            assert pkg in path.resolve().parents, (
                f"{path} lives outside the package and would not be installed")


class TestTheDefaultPrecision:
    def test_it_works_without_x64(self):
        """The suite pins float64; a user will not.

        JAX defaults to float32, and the released package has to be correct
        there too -- so this runs in a subprocess that never sets the flag and
        checks the correction against the value the suite's own mode gives.
        """
        expected = float(model.HmfCorrection().fsigma(1.0, target.FIDUCIAL, 0.5))
        out = _in_subprocess(f"""
import jax, numpy as np
assert not jax.config.jax_enable_x64
from emu_hmf.model import HmfCorrection
print(repr(float(HmfCorrection().fsigma(1.0,
      np.array({list(target.FIDUCIAL)!r}), 0.5))))
""", JAX_ENABLE_X64="0")
        assert float(out) == pytest.approx(expected, rel=1e-6)
