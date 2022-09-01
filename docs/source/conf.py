# Configuration file for the Sphinx documentation builder.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
import furo

#import pathlib
#sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information

project = 'TorchQuantum'
copyright = '2021, Hanrui Wang'
author = 'Hanrui Wang'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon', # google styled docstring
    'sphinx.ext.viewcode', # [source] link to view code
    # 'sphinx.ext.todo',
    # 'sphinx.ext.coverage',
    # 'sphinxcontrib.katex',
    # 'sphinx.ext.autosectionlabel',
    # 'sphinx_copybutton',
    # 'sphinx_panels',
    # 'sphinx.ext.mathjax',
    # 'sphinx.ext.extlinks',
    # 'sphinx_autodoc_typehints',
    # 'jupyter_sphinx',
    'nbsphinx', # support for including Jupyter Notebook (*.ipynb) file
    'recommonmark', # support for including markdown (*.md) file
    # 'sphinx_design',
    # 'sphinx_reredirects'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

# -- Theme
"""
PyTorch

import pytorch_sphinx_theme

# Theme has bootstrap already
panels_add_bootstrap_css = False

#
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#
#

html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    'pytorch_project': 'docs',
    'canonical_url': 'https://pytorch.org/docs/stable/',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'analytics_id': 'UA-117752657-2',
}
"""
"""
Qiskit
import qiskit_sphinx_theme

html_theme = "qiskit_sphinx_theme"
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}
html_static_path = ['_static']
html_css_files = ['custom.css', 'gallery.css']
html_favicon = 'images/favicon.ico'
html_last_updated_fmt = '%Y/%m/%d'
"""
"""
import python_docs_theme
html_theme = 'python_docs_theme'
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': False,
}
"""

html_theme = 'furo'
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': False,
    "index": "page.html"
}

# adjust code block style for readthedocs

# html_static_path = ["_static"]

display_github = False
display_bitbucket = False
display_gitlab = False
show_source = True
# -- Options for EPUB output
epub_show_urls = 'footnote'