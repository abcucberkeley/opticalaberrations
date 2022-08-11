import logging
import sys
import time
import multiprocessing as mp

from typing import Optional
from functools import partial
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve as cupyx_fftconvolve
from scipy.signal import fftconvolve as scipy_fftconvolve
from skimage.util import view_as_windows
from skimage.restoration import richardson_lucy

import quilt3 as q3
from tifffile import imsave, imread

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

classes = {k: label for k, label, in
           zip([
               'Cells',
               'Nuclei',
               'Transmitted Light',
               'Microtubules',
               'Actin Filaments',
               'Desmosomes',
               'DNA',
               'Nucleoli',
               'Nuclear Envelope',
               'Membrane (cellmask)',
               'Membrane (rfp safe harbor)',
               'Actomyosin Bundles',
               'Endoplasmic Reticulum',
               'Golgi Apparatus',
               'Mitochondria'
               ], range(15))
           }


def parse_args(args):
    parser = cli.argparser()

    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="dtype"
    )
    subparsers.required = True

    download = subparsers.add_parser("download")
    download.add_argument('--savedir', default='/media/supernova/data/allencell', type=Path)

    decon = subparsers.add_parser("decon")
    decon.add_argument('psf', type=Path)
    decon.add_argument('samples', type=Path)
    decon.add_argument('--label', type=str, default=None)
    decon.add_argument('--savedir', default='../dataset/allencell/deconvolved', type=Path)

    conv = subparsers.add_parser("conv")
    conv.add_argument('kernels', type=Path)
    conv.add_argument('samples', type=Path)
    conv.add_argument('--savedir', default='../dataset/allencell/convolved', type=Path)

    return parser.parse_args(args)


def download_data(savedir: Path):
    b = q3.Bucket("s3://allencell")
    collection = "aics/label-free-imaging-collection/cells_3d/"
    save_path = Path(f'{savedir}/{collection}/')
    save_path.mkdir(exist_ok=True, parents=True)
    b.fetch(collection, f'{savedir}/{collection}/')

    # for i in b.ls(collection):
    #     for f in tqdm(i):
    #         b.fetch(f['Key'], f"P{savedir}/{f['Key']}")


def deconvolve(sample, psf, label, niters, save_path):
    psf = imread(psf)
    # logger.info(f"PSF: {psf.shape}")
    data = imread(sample)[label]
    # logger.info(f"Data: {data.shape}")
    deconvolved = richardson_lucy(data, psf, niters)
    # logger.info(f"Deconvolved: {deconvolved.shape}")
    imsave(save_path/sample.name, deconvolved)


def convolve(kernel, sample, sample_voxel_size, psf_shape, save_path, cuda=False):
    modelgen = SyntheticPSF(
        n_modes=60,
        lam_detection=.605,
        dtype='confocal',
        psf_shape=psf_shape,
        x_voxel_size=.29,
        y_voxel_size=.29,
        z_voxel_size=.29,
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
    center = [(i // 2) + 1 for i in conv.shape]
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


def create_sample(
    sample: Path,
    kernels: Path,
    savedir: Path,
    sample_voxel_size: tuple,
    psf_shape: tuple
):
    data = imread(sample)
    logger.info(f"Sample: {data.shape}")

    utils.multiprocess(
        partial(
            convolve,
            sample=data,
            psf_shape=psf_shape,
            sample_voxel_size=sample_voxel_size,
            save_path=savedir/sample.stem,
            cuda=True
        ),
        jobs=list(kernels.rglob('*.tif')),
        cores=-1
    )


def create_dataset(
    savedir: Path,
    kernels: Path,
    samples: Path,
    sample_voxel_size: tuple = (.29, .29, .29),
    psf_shape: tuple = (64, 64, 64)
):
    for s in samples.rglob('*.tiff'):
        logger.info(s)
        savedir.mkdir(exist_ok=True, parents=True)

        create_sample(
            sample=s,
            kernels=kernels,
            savedir=savedir,
            sample_voxel_size=sample_voxel_size,
            psf_shape=psf_shape
        )


def decon_dataset(
    savedir: Path,
    psf: Path,
    samples: Path,
    niters: int = 1,
    label: Optional = None,
):
    if label is None:
        for label, k in classes.items():
            logger.info(label)
            save_path = Path(f'{savedir}/{label.lower().replace(" ", "_")}')
            save_path.mkdir(exist_ok=True, parents=True)

            utils.multiprocess(
                partial(
                    deconvolve,
                    psf=psf,
                    label=k,
                    save_path=save_path,
                ),
                jobs=list(samples.rglob('*.tiff')),
                cores=-1
            )
    else:
        k = classes.get(label)
        logger.info(f"{label}: {k}")
        save_path = Path(f'{savedir}/{label.lower().replace(" ", "_")}')
        save_path.mkdir(exist_ok=True, parents=True)

        utils.multiprocess(
            partial(
                deconvolve,
                psf=psf,
                label=k,
                niters=niters,
                save_path=save_path,
            ),
            jobs=list(samples.rglob('*.tiff')),
            cores=-1
        )


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    mp.set_start_method('spawn', force=True)

    if args.dtype == 'conv':
        create_dataset(savedir=args.savedir, kernels=args.kernels, samples=args.samples)

    elif args.dtype == 'decon':
        decon_dataset(savedir=args.savedir, psf=args.psf, samples=args.samples, label=args.label)

    elif args.dtype == 'download':
        download_data(savedir=args.savedir)

    else:
        logger.error("Error: unknown action!")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
