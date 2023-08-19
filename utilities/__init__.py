import os

import toml


def get_version_from_pyproject():
    try:
        # Determine the path to pyproject.toml relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pyproject_path = os.path.join(base_dir, "..", "pyproject.toml")

        # Load the pyproject.toml file
        pyproject_data = toml.load(pyproject_path)

        # Extract the version from the project section
        version = pyproject_data["project"]["version"]
    except FileNotFoundError:
        version = "unknown"
    return version


__version__ = get_version_from_pyproject()
