# Configuration file for the Sphinx documentation builder.

import os
import sys
import time

# -- Path setup --------------------------------------------------------------

# Add project root so autodoc can find the package
try:
    import so4gp
except ImportError:
    sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "so4gp"
author = "Dickson Owuor"
copyright = f"{time.localtime().tm_year}, Dickson Owuor"

# Full version
release = "0.2.5"

# Short version
version = "0.2"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_github_changelog",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "myst_parser",
]

autosummary_generate = True
add_module_names = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

## Enable dollarmath parsing inside Markdown files
#myst_enable_extensions = [
#    "dollarmath",
#    "amsmath",
#]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings --------------------------------------------------------

autodoc_typehints = "description"
autoclass_content = "both"
autodoc_preserve_defaults = True

# Remove duplicate constructor documentation
def remove_lines_before_parameters(app, what, name, obj, options, lines):
    if what == "class":
        first_idx = next(
            (i for i, line in enumerate(lines) if line.startswith(":param")),
            None,
        )
        if first_idx is not None:
            lines[:] = lines[first_idx:]


def setup(app):
    app.connect("autodoc-process-docstring", remove_lines_before_parameters)


# -- HTML output -------------------------------------------------------------

html_theme = "furo"

html_title = "so4gp — Gradual Pattern Mining Algorithms"

html_baseurl = "https://so4gp.readthedocs.io/"

html_copy_source = False

html_theme_options = {
    # "description": "A collection of gradual pattern mining algorithms and tools",  ## Not supported by Furo
    "source_repository": "https://github.com/owuordickson/gp-mining",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
}

html_static_path = ["_static"]
html_css_files = []

# -- EPUB --------------------------------------------------------------------

# epub_show_urls = "footnote"
