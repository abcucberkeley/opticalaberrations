#   to (re)build image, run container:
# docker rm ml_cont || DOCKER_BUILDKIT=0 docker build . -t ml --build-arg BRANCH_NAME=$(git branch --show-current) --progress=plain && docker run -it --name ml_cont --gpus all   ml  /bin/bash

#   to (re)build image with no cache:
# docker rm ml_cont || DOCKER_BUILDKIT=0 docker build . -t ml --build-arg BRANCH_NAME=$(git branch --show-current) --progress=plain --no-cache && docker run -it --name ml_cont --gpus all   ml  /bin/bash

#   to rebuild and tensorflow gpu test
# docker rm ml_cont || DOCKER_BUILDKIT=0 docker build . -t ml --build-arg BRANCH_NAME=$(git branch --show-current) ; docker run --rm --gpus all ml  "~/miniconda3/envs/ml/bin/python -m pytest -vvv --disable-warnings tests/test_tensorflow.py"; docker run -it --name ml_cont --gpus all   ml  /bin/bash


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
# install github self-hosted runner: https://github.com/abcucberkeley/opticalaberrations/settings/actions/runners/new?arch=x64&os=linux
# make github self-hosted runner as a service: https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service
# docker system prune

# test tensorflow GPU:
# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 'conda install tensorflow-gpu' will not install GPU version because GPU is not detected during 'docker build' unless DOCKER_BUILDKIT=0

# Alternative starting points
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04   
# ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/lib64/stubs/:$LD_LIBRARY_PATH

# FROM nvcr.io/nvidia/tensorrt:23.10-py3
# Found tensorflow==2.10 https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html  here: FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3

FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
# docker run --rm -it -gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:23.10-tf2-py3 /bin/bash

# Make sure we have nvidia gpus available during Docker Build or else tensorflow-gpu won't install
# RUN nvidia-smi

# install needed utils. Don't "apt-get upgrade" or else all the NVIDIA tools and drivers will update.
# RUN apt-get update \
#   && apt-get install -y --no-install-recommends \
#     ca-certificates \
#     curl \
#     git \
#     gnupg \
#     wget \
#     vim \
#   #  g++-11 \
#     && rm -rf /var/lib/apt/lists/*


# install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && rm -rf /var/lib/apt/lists/*

# # install miniconda, create ml conda environment with tensorflow-gpu.
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh \
#     && echo "Running $(conda --version)" && \
#     conda init bash && \
#     . /root/.bashrc && \ 
#     conda create -n ml pip setuptools tensorflow-gpu=2.10 python=3.10 --yes -c conda-forge && \
#     conda clean --all --yes


# Download the current github commit page for this branch. This will invalidate the cache for later Docker layers when the commit changes things.
ARG BRANCH_NAME
ADD "https://api.github.com/repos/abcucberkeley/opticalaberrations/commits?sha=${BRANCH_NAME}&per_page=1" dummy_location

# RUN echo "Make sure GPU is active." && nvidia-smi
# git clone the repo, branch=develop, --filter=blob:none will only download the files in HEAD
# WORKDIR /app
# RUN echo "branch=${BRANCH_NAME}" && git clone -b ${BRANCH_NAME} --filter=blob:none --recurse-submodules https://github.com/abcucberkeley/opticalaberrations.git
# WORKDIR /app/opticalaberrations


# git clone the repo, branch=develop, --filter=blob:none will only download the files in HEAD
WORKDIR /docker_install
RUN echo "branch=${BRANCH_NAME}" && git clone -n -b ${BRANCH_NAME} --depth 1 https://github.com/abcucberkeley/opticalaberrations.git 
WORKDIR /docker_install/opticalaberrations
RUN git checkout HEAD requirements.txt

RUN pip install --no-cache-dir -r requirements.txt  --progress-bar off 
# # RUN echo "Running $(conda --version).  Time to update 'ml' environment with yml file. " && conda env update --file win_or_ubuntu_gpu.yml  && conda clean --all --yes

# RUN python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

WORKDIR /app
# SHELL ["/bin/bash", "-l", "-c"]
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]