import logging
import sys
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve as cupyx_fftconvolve
from scipy.signal import fftconvolve as scipy_fftconvolve
from skimage.util import view_as_windows

import quilt3 as q3
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

classes = {k: label for k, label, in
           zip(range(15),
               [
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
               ])
           }


def parse_args(args):
    parser = cli.argparser()

    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="dtype"
    )
    subparsers.required = True

    download = subparsers.add_parser("download")
    download.add_argument('--savedir', default='/media/supernova/data/allencell', type=Path)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument('kernels', type=Path)
    dataset.add_argument('samples', type=Path)
    dataset.add_argument('--savedir', default='../dataset/allencell', type=Path)

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


def convolve(kernel, sample, sample_voxel_size, save_path, cuda=False):
    modelgen = SyntheticPSF(
        n_modes=60,
        lam_detection=.605,
        dtype='confocal',
        psf_shape=(64, 64, 64),
        x_voxel_size=.1,
        y_voxel_size=.1,
        z_voxel_size=.3,
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


def create_sample(
    sample: Path,
    kernels: Path,
    savedir: Path,
    label: int,
    sample_voxel_size: tuple,
    conv_voxel_size: tuple,
    psf_shape: tuple
):
    sample = imread(sample)[label]
    logger.info(f"Sample: {sample.shape}")
    rescaled_shape = [round(sample.shape[i] * (sample_voxel_size[i] / conv_voxel_size[i])) for i in range(3)]
    rescaled_shape = tuple(rescaled_shape[i] if rescaled_shape[i] > psf_shape[i] else psf_shape[i] for i in range(3))

    samples = preprocessing.resize(
        sample,
        voxel_size=conv_voxel_size,
        crop_shape=rescaled_shape,
        sample_voxel_size=sample_voxel_size,
    )
    logger.info(f"Rescaled sample  {samples.shape}")

    windows = view_as_windows(samples, window_shape=psf_shape)

    for i in range(0, windows.shape[0], psf_shape[0]):
        for j in range(0, windows.shape[1], psf_shape[1]):
            for k in range(0, windows.shape[2], psf_shape[2]):
                utils.multiprocess(
                    partial(
                        convolve,
                        sample=windows[i, j, k],
                        sample_voxel_size=conv_voxel_size,
                        save_path=savedir / f'{i}-{j}-{k}',
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
    conv_voxel_size: tuple = (.15, .05, .05),
    psf_shape: tuple = (128, 128, 128)
):
    for s in samples.rglob('*.tiff'):
        logger.info(s)
        for k, label in classes.items():
            logger.info(label)
            save_path = Path(f'{savedir}/convolved/{label.lower().replace(" ", "_")}')
            save_path.mkdir(exist_ok=True, parents=True)

            create_sample(
                sample=s,
                label=k,
                kernels=kernels,
                savedir=save_path,
                sample_voxel_size=sample_voxel_size,
                conv_voxel_size=conv_voxel_size,
                psf_shape=psf_shape
            )
        exit()


def main(args=None):
    timeit = time.time()
    args = parse_args(args)

    mp.set_start_method('spawn', force=True)

    if args.dtype == "dataset":
        create_dataset(savedir=args.savedir, kernels=args.kernels, samples=args.samples)

    elif args.dtype == "download":
        download_data(savedir=args.savedir)

    else:
        logger.error("Error: unknown action!")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
