import logging
import sys
import numpy as np
from pathlib import Path
from tifffile import imread, imsave
import fibsem_tools.io as fibsem
from tqdm import tqdm

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

classes = [
    'er_seg',
    'mito_seg',
    'mito-mem_seg',
]

datasets = [
    'jrc_cos7-11',
    'jrc_hela-2',
    'jrc_hela-3',
    'jrc_jurkat-1',
    'jrc_macrophage-2',
]

urls = [f"s3://janelia-cosem-datasets/{d}/{d}.n5" for d in datasets]


def dowload_data(savedir='../data/FIB-SEM', res='s4'):
    for ds, u in zip(datasets, urls):
        for c in tqdm(classes, desc=ds):
            data = fibsem.read_xarray(f"{u}/labels/{c}/{res}", storage_options={'anon': True})
            print(data)
            save_path = Path(f'{savedir}/{res}/{ds}')
            save_path.mkdir(exist_ok=True, parents=True)

            data = data.compute().data
            imsave(save_path/f'{c}.tif', data)


if __name__ == '__main__':
    dowload_data()
