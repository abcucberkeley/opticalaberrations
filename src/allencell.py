import logging
import sys
import time
import shutil
import multiprocessing as mp

from typing import Optional
from functools import partial
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve as cupyx_fftconvolve
from scipy.signal import fftconvolve as scipy_fftconvolve

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
        help="Arguments for specific action.", dest="cmd"
    )
    subparsers.required = True

    download = subparsers.add_parser("download")
    download.add_argument('--savedir', default='/media/supernova/data/allencell', type=Path)

    decon = subparsers.add_parser("split")
    decon.add_argument('samples', type=Path)
    decon.add_argument('--label', type=str, default=None)
    decon.add_argument('--savedir', default='../dataset/allencell/channels', type=Path)

    conv = subparsers.add_parser("conv")
    conv.add_argument('kernels', type=Path)
    conv.add_argument('samples', type=Path)
    conv.add_argument('--savedir', default='../dataset/allencell/convolved', type=Path)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument('--sample', type=Path)
    dataset.add_argument('--kernels', type=Path)
    dataset.add_argument('--savedir', type=Path)

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


def split(sample, label, save_path, axis=1):
    if axis == 0:
        data = imread(sample)[label]
    elif axis == 1:
        data = imread(sample)[:, label]
    elif axis == 2:
        data = imread(sample)[:, :, label]
    else:
        data = imread(sample)[:, :, :, label]

    imsave(save_path/f"{sample.stem}.tif", data)


def split_channels(
    savedir: Path,
    samples: Path,
    label: Optional = None,
):
    if label is None:
        for label, k in classes.items():
            logger.info(label)
            save_path = Path(f'{savedir}/{label.lower().replace(" ", "_")}')
            save_path.mkdir(exist_ok=True, parents=True)

            utils.multiprocess(
                partial(
                    split,
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
                split,
                label=k,
                save_path=save_path,
            ),
            jobs=list(samples.rglob('*.tiff')),
            cores=-1
        )


def convolve(kernel, sample, sample_voxel_size, psf_shape, save_path, sample_name, cuda=False, embeddings=False):
    modelgen = SyntheticPSF(
        n_modes=55,
        lam_detection=.605,
        psf_shape=psf_shape,
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        snr=1000,
        max_jitter=0,
        na_detection=1.0,
    )

    ker = imread(kernel)

    if cuda:
        conv = cupyx_fftconvolve(cp.array(sample), cp.array(ker), mode='full').get()
    else:
        conv = scipy_fftconvolve(sample, ker, mode='full')

    conv /= np.nanmax(conv)

    width = [(i // 2) for i in sample.shape]
    center = [(i // 2) + 1 for i in conv.shape]
    # center = np.unravel_index(np.argmax(conv, axis=None), conv.shape)
    conv = conv[
           center[0] - width[0]:center[0] + width[0],
           center[1] - width[1]:center[1] + width[1],
           center[2] - width[2]:center[2] + width[2],
           ]

    conv = preprocessing.resize(
        conv,
        crop_shape=modelgen.psf_shape,
        voxel_size=modelgen.voxel_size,
        sample_voxel_size=sample_voxel_size,
        debug=f'{save_path}_embeddings' if embeddings else None,
    )

    if embeddings:
        modelgen.embedding(psf=conv, plot=f'{save_path}_embeddings')

    save_path = Path(f'{save_path}/{"/".join(kernel.parts[-7:-1])}/{sample_name}/{kernel.name}')
    save_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(kernel.with_suffix(".json"), save_path.with_suffix(".json"))
    imsave(save_path, conv)


def create_sample(
    sample: Path,
    kernels: Path,
    savedir: Path,
    sample_voxel_size: tuple = (.6, .15, .15),
    psf_shape: tuple = (64, 64, 64)
):
    data = imread(sample)
    utils.multiprocess(
        partial(
            convolve,
            sample=data,
            psf_shape=psf_shape,
            sample_voxel_size=sample_voxel_size,
            save_path=savedir,
            sample_name=sample.stem,
            cuda=True
        ),
        jobs=list(kernels.rglob('*.tif')),
        cores=-1
    )


def convolve_dataset(
    savedir: Path,
    kernels: Path,
    samples: Path,
):
    for s in samples.rglob('*.tif'):
        create_sample(
            sample=s,
            savedir=savedir,
            kernels=kernels,
        )


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    mp.set_start_method('spawn', force=True)

    if args.cmd == 'conv':
        convolve_dataset(savedir=args.savedir, kernels=args.kernels, samples=args.samples)

    elif args.cmd == 'dataset':
        create_sample(sample=args.sample, kernels=args.kernels, savedir=args.savedir)

    elif args.cmd == 'split':
        split_channels(savedir=args.savedir, samples=args.samples, label=args.label)

    elif args.cmd == 'download':
        download_data(savedir=args.savedir)

    else:
        logger.error("Error: unknown action!")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
