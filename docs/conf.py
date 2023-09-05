# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import datetime
import os
import sys

import toml

sys.path.insert(0, os.path.abspath(".."))


def get_project_data():
    try:
        # Determine the path to pyproject.toml relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pyproject_path = os.path.join(base_dir, "..", "pyproject.toml")

        # Load the pyproject.toml file
        pyproject_data = toml.load(pyproject_path)

        # Extract the version from the project section
        metadata = dict(pyproject_data["project"])
    except Exception as e:
        metadata = "unknown"
    return metadata


metadata = get_project_data()
year = datetime.datetime.now().year

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

if isinstance(metadata, dict):
    project = f"GU Orbit Software {metadata['name']}"
    copyright = f"{year}, {metadata['authors'][0]['name']}"
    author = metadata['authors'][0]['name']
    release = metadata["version"]
else:
    raise TypeError(
        "metadata must be a dict. There must be a problem with the pyproject.toml file."
    )

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
add_module_names = False
extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# insegel theme
# furo theme

html_theme = "furo"


html_static_path = ["style"]
html_css_files = ["custom.css"]
