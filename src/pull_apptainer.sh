#!/bin/bash
ARG1=${1:-CUDA_11_8}
echo "This script will pull the CUDA version passed as the first argument ($ARG1), from the available packages (e.g. CUDA_11_8, CUDA12_3, etc)"
echo "https://github.com/abcucberkeley/opticalaberrations/pkgs/container/opticalaberrations"
apptainer pull --force --disable-cache ../develop_$ARG1.sif oras://ghcr.io/abcucberkeley/opticalaberrations:develop_$ARG1_sif