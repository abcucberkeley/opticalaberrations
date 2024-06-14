#   to build image, run container, interactively:
# docker rm -f ml_cont ; docker build . -t ml --build-arg BRANCH_NAME=$(git branch --show-current) --target Torch_CUDA_12_3 --progress=plain && docker run -it --name ml_cont --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -v ${PWD}:/app/opticalaberrations --workdir /app/opticalaberrations  ml /bin/bash

# to build the TF_CUDA_12_3 image: 
# docker build . -t ghcr.io/abcucberkeley/opticalaberrations:develop_TF_CUDA_12_3 --build-arg BRANCH_NAME=$(git branch --show-current) --target TF_CUDA_12_3 --progress=plain
#
# to run on a ubuntu system:
# install docker: https://docs.docker.com/engine/install/ubuntu/
# set docker permissions for non-root: https://docs.docker.com/engine/install/linux-postinstall/ 
# install nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# install github self-hosted runner: https://github.com/abcucberkeley/opticalaberrations/settings/actions/runners/new?arch=x64&os=linux
# make github self-hosted runner as a service: https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service
# docker system prune
# container's user is different than github action's user, so change permissions of folder: sudo chmod 777 /home/mosaic/Desktop/actions-runner/_work -R

# 'conda install tensorflow-gpu' will not install GPU version because GPU is not detected during 'docker build' unless DOCKER_BUILDKIT=0, so we just do pip install of everything. Actually we will just use NVIDIA container that has it all installed already.

# mounting /clusterfs via this plugin DOES NOT WORK
# docker plugin install vieux/sshfs sshkey.source=/c/Users/Mosaic/.ssh/id_rsa
# docker plugin set vieux/sshfs sshkey.source=/c/Users/Mosaic/.ssh/id_rsa
# docker plugin enable vieux/sshfs
# docker volume create -d vieux/sshfs -o sshcmd=thayeralshaabi@login.abc.berkeley.edu:/clustersfs sshvolume
# [-o IdentityFile=/root/.ssh/<key>] [-o port=<port>] [-o <any_sshfs_-o_option> ] 

# NSIGHT NOT WORKING YET, Doesn't show GPU METRICS
# nvidia nsight profiling:
# https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#AllUsersTag


# this works if you want smb mount a network drive.
# docker volume create --driver local --opt type=cifs  --opt device=//10.17.209.7/nvme2  --opt o=username=abcabc,password=********,file_mode=0777,dir_mode=0777  --name nvme2
# then add this mount flag to the docker run command:     -v nvme2:/clusterfs/nvme2

# this works to mount using ssh keys
# to mount clusterfs using ssh keys (1. copy keys from /.ssh on host to /sshkey in container, 2. make mount point and change permissions for local user, 3. sshfs with that key and no user input):
# docker run --rm -it --gpus all --ipc=host --cap-add=SYS_ADMIN --privileged=true --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864  -u 1000 -v ${HOME}/.ssh:/sshkey -v ${PWD}:/app/opticalaberrations  ghcr.io/abcucberkeley/opticalaberrations:develop_TF_CUDA_12_3 /bin/bash
# sudo mkdir /clusterfs; sudo chmod a+wrx /clusterfs/; sudo chown 1000:1000 -R /sshkey/; sshfs thayeralshaabi@login.abc.berkeley.edu:/clusterfs  /clusterfs -o IdentityFile=/sshkey/id_rsa -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null

# this works to make an apptainer version
# docker run --rm kaczmarj/apptainer pull develop_TF_CUDA_12_3.sif docker://ghcr.io/abcucberkeley/opticalaberrations:develop_TF_CUDA_12_3

# Pass in a target when building to choose the TF Image with the version you want: --build-arg BRANCH_NAME=$(git branch --show-current) --target TF_CUDA_12_3
# For github actions, this is how we will build multiple docker images.
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-12.html#rel-23-12

# for CUDA 12.x
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3 as TF_CUDA_12_3
ENV RUNNING_IN_DOCKER=TRUE

# Make bash colorful https://www.baeldung.com/linux/docker-container-colored-bash-output   https://ss64.com/nt/syntax-ansi.html 
ENV TERM=xterm-256color
RUN echo "PS1='\e[97m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

WORKDIR /docker_install

# Install requirements. Don't "apt-get upgrade" or else all the NVIDIA tools and drivers will update.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  sudo \
  htop \
  cifs-utils \
  winbind \
  smbclient \
  sshfs \
  && rm -rf /var/lib/apt/lists/*

# Git-lfs install
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && rm -rf /var/lib/apt/lists/*

# LabVIEW Linux run-time      # https://www.ni.com/en/support/downloads/software-products/download.labview.html#521764
# RUN git checkout HEAD ${LabVIEW_runtime}
WORKDIR /docker_install
ARG LabVIEW_runtime=ni-labview-2024-runtime-engine_24.1.0.49394-0+f242-ubuntu2204_all.deb 
ADD ${LabVIEW_runtime} ${LabVIEW_runtime}
RUN sudo apt install ./${LabVIEW_runtime}; sudo apt update; sudo apt install libncurses5 libopenal1 libxinerama1 libgl1-mesa-glx libglu1-mesa labview-2024-rte -y; rm -rf /var/lib/apt/lists/*

# Make the dockerfile use the current branch (passed in as a command line argument to "docker build")
ARG BRANCH_NAME

# some helpful shortcuts
ENV cloneit="git clone -b ${BRANCH_NAME} --recurse-submodules https://github.com/abcucberkeley/opticalaberrations.git"
RUN echo 'alias repo="cd /app/opticalaberrations/src"' >> ~/.bashrc
RUN echo 'alias cloneit=${cloneit}' >> ~/.bashrc

# Download the current github commit page for this branch. This will invalidate the cache for later Docker layers when the commit changes things.
# ADD "https://api.github.com/repos/abcucberkeley/opticalaberrations/commits?sha=${BRANCH_NAME}&per_page=1" dummy_location_one

# git clone "requirements.txt" into a junk directory, then pip install. --filter=blob:none will only download the files in HEAD
WORKDIR /docker_install
ADD https://raw.githubusercontent.com/abcucberkeley/opticalaberrations/${BRANCH_NAME}/requirements.txt requirements.txt
RUN echo "branch=${BRANCH_NAME}" && git clone -n -b ${BRANCH_NAME} --depth 1 --filter=blob:none https://github.com/abcucberkeley/opticalaberrations.git 
WORKDIR /docker_install/opticalaberrations
RUN git checkout HEAD requirements.txt
# ADD requirements.txt requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt  --progress-bar off  &&  pip cache purge || true

# Our repo location will be /app/opticalaberrations
# You can switch to this location with "repo" alias command
# This location is typically mounted from your local filesystem when doing "docker run" and the -v flag.
# Otherwise run "cloneit" alias command from an interactive terminal to use git to clone the repo
# For GPU dashboard use: nvitop
WORKDIR /app

ARG USERNAME=vscode
ENV USER=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN   groupadd --gid $USER_GID $USERNAME && \
    groupadd --gid 1001 vscode_secondary && \
    useradd -l --uid $USER_UID --gid $USER_GID -G 1001 -m $USERNAME && \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.        
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME || true

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]




# for CUDA 12.x
FROM nvcr.io/nvidia/pytorch:24.02-py3 as Torch_CUDA_12_3
ENV RUNNING_IN_DOCKER=TRUE

# Make bash colorful https://www.baeldung.com/linux/docker-container-colored-bash-output   https://ss64.com/nt/syntax-ansi.html 
ENV TERM=xterm-256color
RUN echo "PS1='\e[97m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

WORKDIR /docker_install

# Install requirements. Don't "apt-get upgrade" or else all the NVIDIA tools and drivers will update.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  sudo \
  htop \
  cifs-utils \
  winbind \
  smbclient \
  sshfs \
  && rm -rf /var/lib/apt/lists/*

# Git-lfs install
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && rm -rf /var/lib/apt/lists/*

# LabVIEW Linux run-time      # https://www.ni.com/en/support/downloads/software-products/download.labview.html#521764
# RUN git checkout HEAD ${LabVIEW_runtime}
WORKDIR /docker_install
ARG LabVIEW_runtime=ni-labview-2024-runtime-engine_24.1.0.49394-0+f242-ubuntu2204_all.deb 
ADD ${LabVIEW_runtime} ${LabVIEW_runtime}
RUN sudo apt install ./${LabVIEW_runtime}; sudo apt update; sudo apt install libncurses5 libopenal1 libxinerama1 libgl1-mesa-glx libglu1-mesa labview-2024-rte -y; rm -rf /var/lib/apt/lists/*


# Make the dockerfile use the current branch (passed in as a command line argument to "docker build")
ARG BRANCH_NAME

# some helpful shortcuts
ENV cloneit="git clone -b ${BRANCH_NAME} --recurse-submodules https://github.com/abcucberkeley/opticalaberrations.git"
RUN echo 'alias repo="cd /app/opticalaberrations/src"' >> ~/.bashrc
RUN echo 'alias cloneit=${cloneit}' >> ~/.bashrc

# Download the current github commit page for this branch. This will invalidate the cache for later Docker layers when the commit changes things.
# ADD "https://api.github.com/repos/abcucberkeley/opticalaberrations/commits?sha=${BRANCH_NAME}&per_page=1" dummy_location_one

# git clone "requirements.txt" into a junk directory, then pip install. --filter=blob:none will only download the files in HEAD
WORKDIR /docker_install
ADD https://raw.githubusercontent.com/abcucberkeley/opticalaberrations/${BRANCH_NAME}/requirements.txt requirements.txt
RUN echo "branch=${BRANCH_NAME}" && git clone -n -b ${BRANCH_NAME} --depth 1 --filter=blob:none https://github.com/abcucberkeley/opticalaberrations.git 
WORKDIR /docker_install/opticalaberrations
RUN git checkout HEAD requirements.txt
RUN pip install --no-cache-dir -r requirements.txt  --progress-bar off  &&  pip cache purge || true

# Our repo location will be /app/opticalaberrations
# You can switch to this location with "repo" alias command
# This location is typically mounted from your local filesystem when doing "docker run" and the -v flag.
# Otherwise run "cloneit" alias command from an interactive terminal to use git to clone the repo
# For GPU dashboard use: nvitop
WORKDIR /app

ARG USERNAME=vscode
ENV USER=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN   groupadd --gid $USER_GID $USERNAME && \
    groupadd --gid 1001 vscode_secondary && \
    useradd -l --uid $USER_UID --gid $USER_GID -G 1001 -m $USERNAME && \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.        
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]




### Target older CUDA
# Need to use google container, because NVIDIA ones don't have python >3.8 and CUDA 11_x.  But this means more installing
# using different requirements file.
# https://cloud.google.com/deep-learning-containers/docs/choosing-container#versions
# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-gpu.2-12.py310 as CUDA_11_8