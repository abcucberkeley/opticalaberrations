# to (re)build image, run container:
# docker rm ml_cont; docker build . -t ml; docker run -it --name ml_cont --gpus all   ml  /bin/bash

# then (optionally) run tests from within the container:
# conda activate ml
# python -m pytest tests/test_ao.py

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && rm -rf /var/lib/apt/lists/*

# git clone the repo, branch=develop
WORKDIR /app
RUN git clone -b develop --recurse-submodules https://github.com/abcucberkeley/opticalaberrations.git
WORKDIR /app/opticalaberrations

# install miniconda and create ml conda environment
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
    conda env create -f ubuntu.yml && \
    conda activate ml && \
    conda clean --all --yes
  
SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-c"]

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD git pull; conda env update --file ubuntu.yml 