import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'todd'
copyright = '2022, Luting Wang'
author = 'Luting Wang'

release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

if os.environ.get('READTHEDOCS') == 'True':
    html_theme = 'sphinx_rtd_theme'
else:
    html_theme = 'python_docs_theme'
