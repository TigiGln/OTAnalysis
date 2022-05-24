# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
# import mock
# MOCK_MODULES = ['numpy', 'traceback', 'argparse', 'shutil.copy', 'webbrowser', 'pandas', 'pandas.core.tools.numeric', 'scipy', 
# 'scipy.interpolate', 'scipy.optimize', 'scipy.signal', 'matplotlib.pyplot', 'matplotlib.figure', 'matplotlib.lines', 
# 'gridspec', 'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui', 'FigureCanvasQTAgg', 'NavigationToolbar2QT', 'Line2D',
# 'struct.unpack_from', 'ZipFile', 'QWidget', 'QFileDialog', 'QFrame', 'QSpinBox', 'QApplication', 'QMenuBar',
# 'QPushButton', 'QRadioButton', 'QHBoxLayout', 'QVBoxLayout', 'QLabel', 'QMessageBox', 'QLineEdit', 'QGridLayout',
# 'QGroupBox', 'QDoubleSpinBox', 'QButtonGroup', 'QComboBox', 'QScrollArea', 'QMainWindow', 'QAction', 'QDialog', 'QCheckBox',
# 'Qt', 'pyqtSignal', 'QEvent', 'QEventLoop', 'QTimer', 'QIcon']
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()
sys.path.insert(0, os.path.abspath('..'))
from otanalysis import controller, view, model, extractor, tests

# -- Project information -----------------------------------------------------

project = 'OTAnalysis'
copyright = '2022, GNU GENERAL PUBLIC LICENSE'
author = 'Thierry GALLIANO (LAI)'

# The full version, including alpha/beta/rc tags
release = '0.2.72'


# -- General configuration ---------------------------------------------------

# The master toctree document.
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "m2r2",
    "sphinx_rtd_theme",
]

autoclass_content = "both"


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
