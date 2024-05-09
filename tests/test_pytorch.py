import logging
import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')
import torch

import warnings
warnings.filterwarnings("ignore")

import pytest


@pytest.mark.run(order=1)
def test_pytorch(kargs):

    logging.info(f'Pytorch version = {torch.__version__}')
    print(f'\n\nPytorch version = {torch.__version__}\n')

    physical_devices = torch.cuda.device_count()

    gpu_workers = torch.cuda.device_count()

    if gpu_workers > 0:
        gpu_model = torch.cuda.get_device_name(0)
    else:
        gpu_model = None

    print(f'Number of active GPUs: {gpu_workers}, {gpu_model}')
    assert gpu_workers > 0
