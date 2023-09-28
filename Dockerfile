#   to (re)build image, run container:
# docker rm ml_cont; docker build . -t ml; docker run -it --name ml_cont --gpus all   ml  /bin/bash

#   to (re)build image with no cache:
# docker rm ml_cont; docker build . -t ml --no-cache; docker run -it --name ml_cont --gpus all   ml  /bin/bash

#   then (optionally) run tests from within the container:
# conda activate ml
# python -m pytest tests/test_ao.py

#   or as a one-liner:
# docker run --rm -it --gpus all ml  "~/miniconda3/envs/ml/bin/python -m pytest -vvv --disable-warnings tests/test_ao.py"
# docker run --rm -it --gpus all ghcr.io/abcucberkeley/opticalaberrations:develop "~/miniconda3/envs/ml/bin/python -m pytest -vvv --disable-warnings tests/test_ao.py"

# to run on a ubuntu system:
# install docker: https://docs.docker.com/engine/install/ubuntu/
# set docker permissions for non-root: https://docs.docker.com/engine/install/linux-postinstall/ 
# install nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04  # looks like conda is able to download everything we need. so don't need this.
FROM ubuntu:22.04

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gnupg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && rm -rf /var/lib/apt/lists/*

# git clone the repo, branch=develop, --filter=blob:none will only download the files in HEAD
WORKDIR /app
RUN git clone -b develop --filter=blob:none --recurse-submodules https://github.com/abcucberkeley/opticalaberrations.git
WORKDIR /app/opticalaberrations

# install miniconda, use libmamba solver for speed, and create ml conda environment
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda install --yes -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \ 
    conda env create -f ubuntu.yml && \
    conda activate ml && \
    conda clean --all --yes
  
# Download the current github commit page to invalidate cache for later layers when the commit version changes.
ADD "https://github.com/abcucberkeley/opticalaberrations/commits/develop/" dummy_location
RUN git pull && \
    conda env update --file ubuntu.yml --prune

LABEL org.opencontainers.image.source=https://github.com/abcucberkeley/opticalaberrations.git \
      org.opencontainers.image.description="Docker image for a transformer network to perform sensorless detection of aberrations in adaptive optics. https://github.com/abcucberkeley/opticalaberrations" \
      org.opencontainers.image.licenses=BSD-2-Clause 

SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-l", "-c"]
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]