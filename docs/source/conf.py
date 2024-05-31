# pylint: disable=invalid-name

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'todd'
project_copyright = '2022, Luting Wang'
author = 'Luting Wang'

release = '0.5.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

default_role = 'any'

html_static_path = ['_static']
templates_path = ['_templates']

if os.environ.get('READTHEDOCS') == 'True':
    html_theme = 'sphinx_rtd_theme'
else:
    html_theme = 'python_docs_theme'

autodoc_typehints = 'description'
autodoc_default_options = {
    'imported-members': True,
    'members': True,
    'show-inheritance': True,
    'undoc-members': True,
}
autodoc_class_signature = 'separated'
autodoc_inherit_docstrings = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.11', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/master/', None),
    'python-pptx': ('https://python-pptx.readthedocs.io/en/latest/', None),
}
