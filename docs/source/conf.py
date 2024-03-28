# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mosaicperm'
copyright = '2024, Asher Spector'
author = 'Asher Spector'

### Autoversioning
# Import the right package!
import sys
import os
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../../'))

# The full version, including alpha/beta/rc tags
import mosaicperm
release = mosaicperm.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    #'sphinx_multiversion',
    # "sphinx_immaterial",
    # automatic documentation
    ## Option 1:
    # ensures one page per class/function
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    # Option 2: just use autosummary from autodoc
]

# This is needed for automodapi to prevent redundancy
numpydoc_show_class_members = False

# This causes autosummary to generate a separate .rst file for 
# each autosummary directive
autosummary_generate = True

# This changes the code style
pygments_style = 'colorful'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Disable typehints
# autodoc_typehints = "none"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Furi theme
html_theme = 'furo'
html_title = f"v{release}"
html_theme_options = {
    "light_logo": "logo_light.png",
    "dark_logo": "logo_dark.png",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
