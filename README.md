![example branch parameter](https://github.com/guorbit/utilities/actions/workflows/python-app.yml/badge.svg?branch=main) [![codecov](https://codecov.io/github/guorbit/utilities/branch/main/graph/badge.svg?token=3RVZAHQ4W2)](https://codecov.io/github/guorbit/utilities)

![example branch parameter](https://github.com/guorbit/utilities/actions/workflows/python-app.yml/badge.svg?branch=development)

Note before installation: None of these commands have been properly tested. Make sure you installed the package in a virtual environment.

## Installation
For installing the utilities repo as a package use the following commands in the terminal:
Note: you need to have a working ssh key to access github from your current machine.
you need to have wheel installed.

```
pip wheel . --no-deps --wheel-dir=dist

```
After build run the following command to install, the built package.
```
pip install .

```
## Preparing for development
### Prerequisites
It is recommended to use a virtual environment for development. To create a virtual environment, with python 3.10 as of now. To create a virtual environment run the following command in the terminal with anaconda:
```
conda create -n <env_name> python=3.10
```
Then activate the virtual environment by running the following command:
```
conda activate <env_name>
```

The rest of the dependencies are in the pyproject.toml file. To install those dependencies run the following command in the terminal:
```
pip install .[dev]
```
You might potentially have to build the project as well. If yes in that case run the following command:
```
pip wheel . --no-deps --wheel-dir=dist
```
Additionally, it is recommended to use certain extensions for development(in case you are using vs code). These extensions are listed as recommended in the utilities.code-workspace file. To install these extensions, open the utilities.code-workspace file in vs code and click on the install button when prompted. This also enables other checking tools like linting and type checking.
### Project branching strategy
The project is using a staging branching combined with feature branching strategy. The main branch is the production branch. The staging branch is the staging branch meant for advanced longer testing. The development branch is the development branch with experimental features and less reliability. The feature branches are used for new feature development. The feature branches are branched off from the development branch. The development branch is branched off from the staging branch. 
At release the development branch is merged into the staging branch. The staging branch is merged into the main branch, if the branches pass sucsessfully the CI pipeline. - Automated build testing is still under considerationd

The following diagram shows the branching strategy:

```
Main ------------------------------------------> 
                                             ^  
                                            /
Staging -----------------------------------/--->
                                        ^
                                       /
Development --------------------------/-------->
                \                    /
                 \-Feature branches-/

```
### Dynamic backend
The project for certain functionalities relies on a switchable/dynamic backend, provided in some cases hardware accelerated computing, or specialised functionalities for frameworks. 
Available backends:
- Tensorflow
- Pytorch

For this purpose exists the backends directory containing the code for the backends.
The structure looks the following:
```
backends
├── __init__.py
├── tf                      <------- Defines implementation for tensorflow backend
│   ├── __init__.py
│   ├── somemodule.py
├── torch                   <------- Defines implementation for pytorch backend
│   ├── __init__.py
│   ├── somemodule.py
├── common                  <------- Defines the interface for the backends
│   ├── __init__.py
│   ├── somemodule.py
```
**Note:** The interfaces are never used directly however it provides structure for the backend implementations, such that it remains the same through all backends.\
**Note:** In order for the dynamic backend to work properly the backends skeleton has to be identical.
### Testing
To run the tests run the following command in the terminal:
```
pytest
```
If the above command doesn't work, run the following command:
```
python -m pytest
```
In case you want to run a coverage test, run the following command:
```
python -m pytest --cov --cov-report term
```

### Linting
To run the linting tests run the following command in the terminal (this also loads in the linting configuration stored in the project root):
```
pylint --rcfile=.pylintrc utilities
```

## Future plans (Ever expanding list)
- [ ] Docker build and deployment for the package, to docker hub.
- [ ] PyPi build and deployment for the package.
- [ ] Additional CI pipeline for staging environment.
- [ ] Mutation testing.
- [ ] Tensorboard building package
- [ ] Scalable ML data reading pipeline
- [ ] Model conversion pipelines
- [ ] Image manupulation tools
- [ ] Image augmentation tools

Submit a suggestion for a new feature by creating an issue, or if you already have it done by creating a pull request to development.
