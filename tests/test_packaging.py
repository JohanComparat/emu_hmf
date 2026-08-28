r"""What ends up inside a built distribution.

The trained weights are the package.  Without a ``package-data`` glob a wheel
installs the code and none of its data, ``HmfCorrection()`` raises
``FileNotFoundError`` on first use, and nothing in a source checkout notices --
every other test in this suite reads the weights straight out of the working
tree.  So the build is run and the archives are opened.
"""
import importlib.util
import pathlib
import subprocess
import sys
import tarfile
import zipfile

import pytest

from emu_hmf import model

ROOT = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if importlib.util.find_spec("build.__main__") is None:
        pytest.skip("`python -m build` is a [dev] extra and is not installed")
    out = tmp_path_factory.mktemp("dist")
    r = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out), str(ROOT)],
        capture_output=True, text=True)
    if r.returncode:
        pytest.fail(f"build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, (wheels, sdists)
    return wheels[0], sdists[0]


class TestTheWheelCarriesTheWeights:
    def test_every_shipped_weights_file_is_inside_it(self, built):
        wheel, _ = built
        with zipfile.ZipFile(wheel) as z:
            names = set(z.namelist())
        for path in model.WEIGHTS.values():
            assert f"emu_hmf/data/{path.name}" in names, (
                f"{path.name} is missing from the wheel; an install would "
                "raise FileNotFoundError on the first call")

    def test_they_are_not_empty(self, built):
        wheel, _ = built
        with zipfile.ZipFile(wheel) as z:
            for path in model.WEIGHTS.values():
                assert z.getinfo(f"emu_hmf/data/{path.name}").file_size > 1000

    def test_the_licence_is_declared_in_the_metadata(self, built):
        wheel, _ = built
        with zipfile.ZipFile(wheel) as z:
            meta = next(n for n in z.namelist() if n.endswith("METADATA"))
            text = z.read(meta).decode()
        assert "MIT" in text
        assert "License-File: LICENSE" in text


class TestTheSdistIsBuildable:
    def test_it_carries_the_licence_the_tests_and_the_weights(self, built):
        _, sdist = built
        with tarfile.open(sdist) as t:
            names = {n.split("/", 1)[1] for n in t.getnames() if "/" in n}
        assert "LICENSE" in names
        assert "README.md" in names
        assert "pyproject.toml" in names
        assert any(n.startswith("tests/") for n in names)
        for path in model.WEIGHTS.values():
            assert f"emu_hmf/data/{path.name}" in names


class TestAFreshInstallWorks:
    def test_the_wheel_installs_and_evaluates_away_from_the_source(
            self, built, tmp_path):
        """Installed into an empty prefix and run from a directory that is not
        the repository, so nothing can be picked up from the working tree."""
        wheel, _ = built
        target_dir = tmp_path / "site"
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
             "--target", str(target_dir), str(wheel)],
            capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"pip could not install into a target dir: {r.stderr}")
        import os

        env = dict(os.environ, PYTHONPATH=str(target_dir))
        env.pop("JAX_PLATFORMS", None)
        r = subprocess.run(
            [sys.executable, "-c",
             "import emu_hmf, numpy as np;"
             "from emu_hmf.model import HmfCorrection, WEIGHTS;"
             "from emu_hmf.target import FIDUCIAL;"
             "print(emu_hmf.__file__);"
             "print([float(HmfCorrection(WEIGHTS[k]).fsigma(1.0, FIDUCIAL, 0.5))"
             " for k in sorted(WEIGHTS)])"],
            capture_output=True, text=True, cwd=str(tmp_path), env=env)
        assert r.returncode == 0, r.stderr
        used, values = r.stdout.strip().splitlines()[-2:]
        assert str(ROOT) not in used, f"it imported the source tree: {used}"
        assert all(v > 0 for v in eval(values)), values
