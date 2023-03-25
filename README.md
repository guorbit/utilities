![example branch parameter](https://github.com/guorbit/utilities/actions/workflows/python-app.yml/badge.svg?branch=main) [![codecov](https://codecov.io/github/guorbit/utilities/branch/main/graph/badge.svg?token=3RVZAHQ4W2)](https://codecov.io/github/guorbit/utilities)

Note before installation: None of these commands have been properly tested. Make sure you installed the package in a virtual environment.

For installing the utilities repo as a package use the following commands in the terminal:
Note: you need to have a working ssh key to access github from your current machine.

```
pip install git+ssh://git@github.com:guorbit/utilities.git

```


Alternatively the following command can be used to install a git repo AFTER cloning it:
Note: the path below has to be modified to point to the package directory.
```
pip install git+file:///path/to/your/package#egg=package-name

```
