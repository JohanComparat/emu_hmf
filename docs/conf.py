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
    "sphinx_copybutton",
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

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"emu_hmf {release}"

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
master_doc = "index"

# ``\dd`` is not a MathJax primitive -- it comes from LaTeX's `physics` package,
# which is not loaded here -- so every ``\dd`` in this documentation rendered as
# a red "Undefined control sequence" on the published site.  Twenty-six of them,
# across six pages, including the two central equations: the ratio being fitted
# and the abundance.
#
# ``sphinx -W`` cannot catch this.  MathJax runs in the reader's browser, long
# after the build has succeeded, so the only way to see it is to look at a
# rendered page.  Defining the macro is the fix; keeping the sources as ``\dd``
# keeps them readable as LaTeX.
mathjax3_config = {
    "tex": {
        "macros": {
            "dd": r"\mathrm{d}",
        },
    },
}

# `-W` is on in CI, and `nitpicky` is what gives it something to catch.  With
# it off, a Python cross-reference to a name that does not exist emits no
# warning at all -- so `-W` passes and the link is simply dead.
#
# It was off, and the comment here claimed the opposite.  What slipped through
# was a `:data:` reference to a constant that had been renamed, on the page
# documenting the release that renamed it.
nitpicky = True

# Names that legitimately have no target: the standard library and third-party
# packages this one deliberately does not depend on, plus the type names that
# appear in signatures rather than in the API.
nitpick_ignore_regex = [
    (r"py:class", r"^(numpy|np|jax|jnp|optax|types)\..*"),
    (r"py:class", r"^(array_like|ArrayLike|Array|optional|pathlib\.Path)$"),
    # Sibling packages this one deliberately does not depend on.  The bare
    # module name as well as dotted paths: ``:mod:`emu_pk``` has no dot and
    # would otherwise slip past the pattern and fail the build.
    (r"py:.*", r"^(ggah_mod|emu_pk|CEmulator|classy|optax)(\..*)?$"),
    # `ggah_mod`'s `Cosmology` fields, written unqualified because the prose
    # reads better that way and because this package documents the *convention*
    # rather than the class.  They have no target here by construction: the
    # dependency is deliberately one-way and lazy.
    (r"py:(attr|class)",
     r"^(Cosmology|Omega_m|Omega_b|Omega_cb|Omega_cdm|Omega_nu|Omega_nu_matter"
     r"|Omega_k|rho_cold|f_nu|ln10A_s|sum_mnu|nu_hierarchy|n_s|w0|wa|h)$"),
]
