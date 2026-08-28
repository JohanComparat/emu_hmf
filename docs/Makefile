SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html strict figures clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

strict:
	@$(SPHINXBUILD) -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

# Regenerate the committed figures.  Not part of `html`: the documentation
# builder has no matplotlib, and the PNGs are in the repository.
figures:
	@python make_figures.py

clean:
	rm -rf "$(BUILDDIR)"
