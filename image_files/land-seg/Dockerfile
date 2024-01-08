FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
RUN yes| unminimize
# Set ENV variables
ENV LANG C.UTF-8
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
ENV GIT_CLONE="git clone --depth 10"

# installing base operation packages

RUN $APT_INSTALL software-properties-common

RUN $APT_INSTALL curl

RUN $APT_INSTALL git

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
# Installing python3.11
    $APT_INSTALL \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-distutils-extra

# Add symlink so python and python3 commands use same python3.10 executable
RUN ln -s /usr/bin/python3.10 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python
# Installing pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
ENV PATH=$PATH:/root/.local/bin

# Update setuptools
RUN pip install --upgrade setuptools

# Installing pip packages


RUN $PIP_INSTALL install git+https://github.com/guorbit/utilities.git@development

RUN $PIP_INSTALL torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

RUN $PIP_INSTALL \
    cython\
    torcheval==0.0.7\
    scikit-learn==1.3.2 \
    numpy==1.23.4 \
    pandas==1.5.0 \
    matplotlib==3.6.1 \
    ipython==8.5 \
    ipykernel==6.16.0 \
    ipywidgets==8.0.2 \
    tqdm==4.64.1 \
    pillow==9.2.0 \
    seaborn==0.12.0 \
    tabulate==0.9.0 \
    jsonify==0.5 \
    wandb==0.13.4 \
    jupyterlab-snippets==0.4.1  \
    boto3==1.33.0 \
    pytorch-msssim==1.0.0 \
    einops==0.7.0 \
    notebook==6.5.5 \
    timm==0.9.12 \
    omegaconf==2.3.0 \
    xformers==0.0.18 \
    fvcore \
    yapf \
    addict \
    openmim

RUN mim install mmcv-full==1.5 --no-deps
RUN mim install mmengine --no-deps
RUN pip install mmsegmentation==0.27.0
    


RUN curl -sL https://deb.nodesource.com/setup_16.x | bash  && \
    $APT_INSTALL nodejs  && \
    $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git && \
    jupyter contrib nbextension install --user




EXPOSE 8888 6006
CMD jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True

# Install dependencies
