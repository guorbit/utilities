![example branch parameter](https://github.com/guorbit/utilities/actions/workflows/python-app.yml/badge.svg?branch=main) [![codecov](https://codecov.io/github/guorbit/utilities/branch/main/graph/badge.svg?token=3RVZAHQ4W2)](https://codecov.io/github/guorbit/utilities)

Note before installation: None of these commands have been properly tested. Make sure you installed the package in a virtual environment.

For installing the utilities repo as a package use the following commands in the terminal:
Note: you need to have a working ssh key to access github from your current machine.
you need to have wheel installed.

```
python setup.py bdist_wheel sdist

```
After build run the following command to install, the built package.
```
pip install .

```
