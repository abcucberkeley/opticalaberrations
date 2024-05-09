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

    gpu_workers = torch.cuda.device_count()

    if gpu_workers > 0:
        for i in range(gpu_workers):
            gpu_model = torch.cuda.get_device_name(i)
            logging.info(f'Number of active GPUs: {gpu_workers}, {gpu_model}')
        logging.info(f'GPU memory free : {torch.cuda.mem_get_info()[0] / (1024*1024*1024):0.1f} GB')
        logging.info(f'GPU memory total: {torch.cuda.mem_get_info()[1] / (1024*1024*1024):0.1f} GB')
    else:
        gpu_model = None
        logging.info(f'Number of active GPUs: {gpu_workers}, {gpu_model}')


    assert gpu_workers > 0
