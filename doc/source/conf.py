# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath("../../"))
project = 'Continuous data REPresentation'
author = 'Vincent Laurent; Fabien Pavy'
release = '2024.2.5'



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'myst_parser',
              "sphinx.ext.coverage",
              'sphinx.ext.todo',
              'sphinx_copybutton',
              'sphinx_favicon',
              'autoapi.extension',
              'versionwarning.extension',
              "sphinx-prompt",
              'sphinx.ext.coverage',
              'numpydoc',
              'nbsphinx',
              ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# html_style = '_static/palma.css'

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autoapi_dirs = [
    '../../crep',
]
autoapi_ignore = ["*table.py*"]
