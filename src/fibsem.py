import logging
import sys
import time
from pathlib import Path
import fibsem_tools.io as fibsem
import xarray
from tifffile import imsave
from tqdm import tqdm

import cli

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
        help="Arguments for specific action.", dest="dtype"
    )
    subparsers.required = True

    download = subparsers.add_parser("download")
    download.add_argument('--savedir', default='../data/FIB-SEM', type=Path)
    download.add_argument('--resolution', default='s3', type=str)
    download.add_argument('--format', default='zarr', type=str)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument('kernels', type=Path)
    dataset.add_argument('samples', type=Path)
    dataset.add_argument('--savedir', default='../dataset/FIB-SEM', type=Path)

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
                    data.to_zarr(save_path / f'{c}.zarr', mode='w', compute=True)
                    logger.info(data)
                else:
                    data = data.compute().data
                    imsave(save_path/f'{c}.tif', data)

                logger.info(data)

            except Exception as e:
                logger.warning(f'`{c}` not found for `{ds}`')


def create_dataset(savedir: Path, kernels: Path, samples: Path):
    samples = xarray.open_zarr(samples)
    print(samples)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)

    if args.dtype == "dataset":
        create_dataset(savedir=args.savedir, kernels=args.kernels, samples=args.samples)

    elif args.dtype == "download":
        download_data(savedir=args.savedir, resolution=args.resolution)

    else:
        logger.error("Error: unknown action!")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
