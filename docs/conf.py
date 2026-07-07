"""Sphinx configuration for the DoE documentation.

Build with::

    uv run --extra docs sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

from importlib.metadata import version as _version

project = "DoE"
author = "Scott Warchal"
copyright = "2026, Scott Warchal"
release = _version("doe")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google-style ``Args:`` / ``Returns:`` docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Markdown (MyST) alongside reStructuredText, so the narrative docs stay as .md.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# VIGNETTES.md links to this pre-rendered run-sheet example; copy it into the site root.
html_extra_path = ["example_design.html"]

html_theme = "furo"
html_title = f"DoE {release}"
