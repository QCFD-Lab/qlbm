# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.append(os.path.abspath("../../.."))
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))

project = "qlbm"
copyright = "2024, qlbm authors"
author = "qlbm authors"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

autodoc_member_order = "bysource"

html_theme_options = {
    "secondary_sidebar_items": {
        "**": ["page-toc", "searchbox", "sourcelink"],
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/qcfd-lab/qlbm",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/qlbm/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "collapse_navigation": True,
    # "show_nav_level": 5,
    # "navigation_depth": 5,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}

html_context = {
    "github_url": "https://github.com",
    "github_user": "qcfd-lab",
    "github_repo": "qlbm",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

master_doc = "index"
