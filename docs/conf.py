import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath("../src"))

project = "hdvec"
author = "Brodie Schroeder, Twistient Corp."
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"
exclude_patterns = [
    "_build",
]
