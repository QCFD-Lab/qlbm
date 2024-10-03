# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib
import inspect
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    "sphinx.ext.linkcode",
    "sphinx_favicon",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "label"

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
    "github_version": "main/docs",
    "display_github": True,
}

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

master_doc = "index"

# always_document_param_types = True
# typehints_fully_qualified = True

autodoc_typehints = "both"
autodoc_typehints_format = "short"

favicons = [
    "favicon-16x16.png",
    "favicon-32x32.png",
]


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    module_name = info["module"]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None:
        return None
    try:
        file_name = info["module"].replace(".", "/")
    except ValueError:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
        return None
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"

    return f"https://github.com/qcfd-lab/qlbm/tree/main/{file_name}.py{linespec}"
