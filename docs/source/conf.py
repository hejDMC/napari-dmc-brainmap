# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'napari-dmc-brainmap'
copyright = '2025, DMC'
author = 'DMC'
release = '0.1.7b6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx.ext.autosummary",  # Generates function/method summaries
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


autosummary_generate = True  # Automatically generate stub files for functions
napoleon_google_docstring = True  # Enable Google-style docstrings
napoleon_numpy_docstring = True   # Enable NumPy-style docstrings

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['']