# backends/__init__.py
from . import tf, torch


def set_backend(name):
    """
    Provides an interface to set the backend.

    Parameters
    ----------
    :param name: The name of the backend to set.

    Example
    -------
    >>> from utilities.backends import set_backend
    >>> set_backend('tf') # or 'torch'
    >>> from utilities.backends import backend
    >>> somefile_module = backend.somefile
    >>> somefile_module.some_fun()
    out: "hello from backend"

    Available backends
    ------------------
    * tf
    * torch
    """
    print(f"Setting backend to {name}")
    global backend
    if name == 'tf':
        backend = tf
    elif name == 'torch':
        backend = torch
    else:
        raise ValueError(f"Unknown backend: {name}")


# Set a default backend
set_backend('tf')
