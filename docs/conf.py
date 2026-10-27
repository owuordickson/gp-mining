"""Configuration for the Sphinx documentation builder."""

import sys
import time
from pathlib import Path

# Document the source checkout, even when a different release is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from so4gp import __version__  # noqa: E402

project = "so4gp"
author = "Dickson Owuor"
copyright = f"{time.localtime().tm_year}, Dickson Owuor"
release = __version__
version = ".".join(release.split(".")[:2])

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
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath"]
exclude_patterns = ["_build", ".superpowers", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

autodoc_typehints = "description"
autoclass_content = "both"
autodoc_preserve_defaults = True

autodoc_type_aliases = {"pd.DataFrame": "pandas.DataFrame"}

linkcheck_timeout = 10
linkcheck_retries = 0
linkcheck_workers = 5

html_theme = "furo"
html_title = "so4gp — Gradual Pattern Mining Algorithms"
html_baseurl = "https://so4gp.readthedocs.io/"
html_copy_source = False
html_theme_options = {
    "source_repository": "https://github.com/owuordickson/gp-mining",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
}
html_static_path = ["_static"]
