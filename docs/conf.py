# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source directory to the path for autodoc
sys.path.insert(0, os.path.abspath("../packages/moreau/python"))

# -- Project information -----------------------------------------------------
project = "Moreau"
copyright = "2026, The Moreau Project"
author = "The Moreau Project"
release = "0.4.0-beta.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

# MyST settings for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]

# Generate anchors for h1-h3 headings so cross-doc #slug links resolve
myst_heading_anchors = 3

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # Don't expose undocumented internals
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Hide private members from documentation
autodoc_default_flags = ["members"]

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping for cross-references to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Source file settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_extra_path = ["CNAME"]
html_title = "Moreau Docs"
html_favicon = "_static/favicon.svg"
html_logo = None

# Furo theme options
html_theme_options = {
    "announcement": (
        "Moreau is open source — "
        "<a href='https://github.com/moreau-project/moreau'>star us on GitHub</a> "
        "or <code>pip install moreau</code>"
    ),
    "source_repository": "https://github.com/moreau-project/moreau",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#0066cc",  # Blue
        "color-brand-content": "#0066cc",
        "color-admonition-background": "#f8fafc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4da6ff",
        "color-brand-content": "#4da6ff",
        "color-admonition-background": "#1e293b",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "footer_icons": [],
}

# Pygments syntax highlighting
pygments_style = "default"
pygments_dark_style = "monokai"

# Custom CSS
html_css_files = ["custom.css"]

# Suppress specific warnings
suppress_warnings = ["autodoc.import_object"]
