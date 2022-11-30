import logging
import sys
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path


import fibsem_tools.io as fibsem
import numpy as np
import xarray as xr
import cupy as cp
from cupyx.scipy.signal import fftconvolve as cupyx_fftconvolve
from scipy.signal import fftconvolve as scipy_fftconvolve
from skimage.util import view_as_windows

from tifffile import imsave, imread
from tqdm import tqdm, trange

import utils
import cli
import preprocessing
from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

classes = [
    'cent_seg',
    'cent-dapp_seg',
    'chrom_seg',
    'ecs_seg',
    'endo_seg',
    'endo-mem_seg',
    'er_seg',
    'er-mem_seg',
    'mito_seg',
    'nucleus_seg',
    'pm_seg',
    'vesicle_seg'
]

datasets = [
    'jrc_hela-1',
    'jrc_hela-2',
    'jrc_hela-3',
    'jrc_jurkat-1',
    # 'jrc_cos7-11',
    # 'jrc_sum159-1',
    # 'jrc_macrophage-2',
    # 'jrc_choroid-plexus-2',
]

urls = [f"s3://janelia-cosem-datasets/{d}/{d}.n5" for d in datasets]


def parse_args(args):
    parser = cli.argparser()

    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="cmd"
    )
    subparsers.required = True

    download = subparsers.add_parser("download")
    download.add_argument('--savedir', default='../data/FIB-SEM', type=Path)
    download.add_argument('--resolution', default='s3', type=str)
    download.add_argument('--format', default='zarr', type=str)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument('kernels', type=Path)
    dataset.add_argument('samples', type=Path)
    dataset.add_argument('--savedir', default='../data/FIB-SEM', type=Path)

    return parser.parse_args(args)


def download_data(savedir: Path, resolution: str, dtype: str = 'zarr'):
    for ds, u in zip(datasets, urls):
        for c in tqdm(classes, desc=ds):
            try:
                data = fibsem.read_xarray(
                    f"{u}/labels/{c}/{resolution}",
                    storage_options={'anon': True},
                    # chunks=(128, 128, 128)
                )
                save_path = Path(f'{savedir}/{dtype}/{resolution}/{ds}')
                save_path.mkdir(exist_ok=True, parents=True)

                if dtype == 'zarr':
                    data = data.to_dataset()
                    data = data.rename({list(data.keys())[0]: 'data'})
                    data.to_zarr(save_path / f'{c}.zarr', mode='w', compute=True)
                    logger.info(data)
                else:
                    data = data.compute().data
                    imsave(save_path/f'{c}.tif', data)

                logger.info(data)

            except Exception as e:
                logger.warning(f'`{c}` not found for `{ds}`')


def convolve(kernel, sample, sample_voxel_size, save_path, cuda=False):
    modelgen = SyntheticPSF(
        n_modes=55,
        lam_detection=.605,
        psf_shape=(64, 64, 64),
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        snr=100,
        max_jitter=0,
    )

    ker = imread(kernel)

    if cuda:
        conv = cupyx_fftconvolve(cp.array(sample), cp.array(ker), mode='full').get()
    else:
        ker = imread(kernel)
        conv = scipy_fftconvolve(sample, ker, mode='full')

    conv /= np.nanmax(conv)

    width = [(i // 2) for i in sample.shape]
    # center = [(i // 2) + 1 for i in conv.shape]
    center = np.unravel_index(np.argmax(conv, axis=None), conv.shape)
    conv = conv[
       center[0] - width[0]:center[0] + width[0],
       center[1] - width[1]:center[1] + width[1],
       center[2] - width[2]:center[2] + width[2],
    ]
    save_path = f'{save_path}_{"_".join(kernel.parts[-4:])}'
    imsave(save_path, conv)

    conv = preprocessing.resize(
        conv,
        crop_shape=modelgen.psf_shape,
        voxel_size=modelgen.voxel_size,
        sample_voxel_size=sample_voxel_size,
        debug=f'{save_path}_embeddings',
    )

    modelgen.embedding(psf=conv, plot=f'{save_path}_embeddings')


def create_dataset(
    savedir: Path,
    kernels: Path,
    samples: Path,
    sample_voxel_size: tuple = (.032, .032, .032),
    conv_voxel_size: tuple = (.3, .075, .075),
    psf_shape: tuple = (128, 128, 128)
):
    save_path = Path(f'{savedir}/convolved')
    save_path.mkdir(exist_ok=True, parents=True)

    samples = xr.open_dataset(samples)
    logger.info(samples)

    samples = np.array(samples.data.compute())
    samples[samples > 1] = 1.
    rescaled_shape = [round(samples.shape[i] * (sample_voxel_size[i]/conv_voxel_size[i])) for i in range(3)]
    rescaled_shape = tuple(
        rescaled_shape[i] if rescaled_shape[i] > psf_shape[i] else psf_shape[i]
        for i in range(3)
    )

    samples = preprocessing.resize(
        samples,
        voxel_size=conv_voxel_size,
        crop_shape=rescaled_shape,
        sample_voxel_size=sample_voxel_size,
    )
    logger.info(f"Rescaled shape: {samples.shape}")

    windows = view_as_windows(samples, window_shape=psf_shape)
    logger.info(windows.shape)

    for i in range(0, windows.shape[0], psf_shape[0]//2):
        for j in range(0, windows.shape[1], psf_shape[1]//2):
            for k in range(0, windows.shape[2], psf_shape[2]//2):
                utils.multiprocess(
                    partial(
                        convolve,
                        sample=windows[i, j, k],
                        sample_voxel_size=conv_voxel_size,
                        save_path=save_path/f'{i}-{j}-{k}',
                        cuda=True
                    ),
                    jobs=list(kernels.rglob('*.tif')),
                    cores=-1
                )


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    mp.set_start_method('spawn', force=True)

    if args.cmd == "dataset":
        create_dataset(savedir=args.savedir, kernels=args.kernels, samples=args.samples)

    elif args.cmd == "download":
        download_data(savedir=args.savedir, resolution=args.resolution)

    else:
        logger.error("Error: unknown action!")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
