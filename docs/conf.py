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
    author = metadata["authors"][0]["name"]
    release = metadata["version"]
else:
    raise TypeError(
        "metadata must be a dict. There must be a problem with the pyproject.toml file."
    )


def setup(app):
    app.connect("builder-inited", add_jinja_filters)


def add_jinja_filters(app):
    app.builder.templates.environment.filters["extract_last_part"] = extract_last_part


def extract_last_part(fullname):
    return fullname.split(".")[-1]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
add_module_names = False
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "docs.custom_modules.auto_toctree",
]

autodoc_default_options = {
    "members": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# insegel theme
# furo theme

html_theme = "furo"

html_theme_options = {
    "dark_css_variables": {
        "color-api-background": "#202020",
        "color-api-background-hover": "#505050",
        "color-sidebar-item-background--current": "#303030",
        "color-sidebar-item-background--hover": "#303030",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "./tf2_py_objects.inv",
    ),
}

autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented_params"

html_static_path = ["style"]
html_css_files = ["custom.css"]
