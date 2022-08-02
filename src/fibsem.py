import logging
import sys
import time
from pathlib import Path
import fibsem_tools.io as fibsem
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
    download.add_argument('--savedir', default='../data/FIB-SEM')
    download.add_argument('--resolution', default='s3')
    download.add_argument('--format', default='zarr')

    return parser.parse_args(args)


def download_data(savedir, resolution, format='zarr'):
    for ds, u in zip(datasets, urls):
        for c in tqdm(classes, desc=ds):
            try:
                data = fibsem.read_xarray(
                    f"{u}/labels/{c}/{resolution}",
                    storage_options={'anon': True},
                    # chunks=(128, 128, 128)
                )
                save_path = Path(f'{savedir}/{format}/{resolution}/{ds}')
                save_path.mkdir(exist_ok=True, parents=True)

                if format == 'zarr':
                    data = data.to_dataset()
                    data.to_zarr(save_path / f'{c}.zarr', mode='w', compute=True)
                    logger.info(data)
                else:
                    data = data.compute().data
                    imsave(save_path/f'{c}.tif', data)

                logger.info(data)

            except Exception as e:
                logger.warning(f'`{c}` not found for `{ds}`')


def main(args=None):
    timeit = time.time()
    args = parse_args(args)

    if args.dtype == "download":
        download_data(savedir=args.savedir, resolution=args.resolution)

    else:
        logger.error("Error: unknown action!")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
