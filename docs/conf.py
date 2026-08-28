import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import emu_hmf                                                    # noqa: E402

project = "emu_hmf"
author = "Johan Comparat"
copyright = "2026, Johan Comparat"
release = emu_hmf.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
# The generation half imports CLASS and the CSST emulator lazily, inside the
# functions that use them.  Nothing here imports them at module scope, so the
# builder needs no mocks -- but autodoc still resolves annotations, so anything
# that ever moves to a module-level import belongs in this list.
autodoc_mock_imports = []

napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
}
html_static_path = ["_static"]
html_title = f"emu_hmf {release}"

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

# `-W` is on in CI, so a broken cross-reference fails the build rather than
# becoming a dead link nobody notices.
nitpicky = False
