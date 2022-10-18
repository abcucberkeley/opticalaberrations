import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import time
from pathlib import Path
import cli
import experimental


def parse_args(args):
    parser = cli.argparser()
    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="func"
    )
    subparsers.required = True

    deskew = subparsers.add_parser("deskew")
    deskew.add_argument("input", type=Path, help="path to input .tif file")
    deskew.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    deskew.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    deskew.add_argument(
        "--flipz", action='store_true',
        help='a toggle to flip Z axis'
    )

    points = subparsers.add_parser("points")
    points.add_argument("input", type=Path, help="path to input .tif file")
    points.add_argument("psf", type=Path, help="path to PSF .tif file")
    points.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    points.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )

    predict = subparsers.add_parser("predict")
    predict.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict.add_argument("input", type=Path, help="path to input .tif file")
    predict.add_argument("pattern", type=Path, help="path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)")
    predict.add_argument(
        "--state", default=None, type=Path,
        help="optional path to current DM state .csv file (Default: `blank mirror`)"
    )
    predict.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        type=str, help='type of the desired PSF'
    )
    predict.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict.add_argument(
        "--model_lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict.add_argument(
        "--model_axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    predict.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict.add_argument(
        "--scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    predict.add_argument(
        "--threshold", default=1e-2, type=float,
        help='set predictions below threshold to zero (microns)'
    )
    predict.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )

    predict_rois = subparsers.add_parser("predict_rois")
    predict_rois.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_rois.add_argument("input", type=Path, help="path to input .tif file")
    predict_rois.add_argument("peaks", type=Path, help="path to point detection results (.mat file)")
    predict_rois.add_argument("pattern", type=Path, help="path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)")

    predict_rois.add_argument(
        "--state", default=None, type=Path,
        help="optional path to current DM state .csv file (Default: `blank mirror`)"
    )
    predict_rois.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_rois.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat', type=str,
        help='type of the desired PSF'
    )
    predict_rois.add_argument(
        "--window_size", default=32, type=int, help='size of the window to crop around each point of interest'
    )
    predict_rois.add_argument(
        "--num_rois", default=10, type=int, help='max number of detected points to use for estimating aberrations'
    )
    predict_rois.add_argument(
        "--min_intensity", default=200, type=int, help='minimum intensity desired for detecting peaks of interest'
    )
    predict_rois.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_rois.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_rois.add_argument(
        "--model_lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_rois.add_argument(
        "--model_axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    predict_rois.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_rois.add_argument(
        "--scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    predict_rois.add_argument(
        "--threshold", default=1e-2, type=float,
        help='set predictions below threshold to zero (microns)'
    )
    predict_rois.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )

    return parser.parse_args(args)


def main(args=None):

    timeit = time.time()
    args = parse_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger('')
    logger.info(args)

    if args.func == 'deskew':
        experimental.deskew(
            img=args.input,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            flipz=args.flipz,
        )

    elif args.func == 'points':
        experimental.points_detection(
            img=args.input,
            psf=args.psf,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
        )

    elif args.func == 'predict':
        experimental.predict(
            model=args.model,
            img=args.input,
            dm_pattern=args.pattern,
            dm_state=args.state,
            prev=args.prev,
            psf_type=args.psf_type,
            axial_voxel_size=args.axial_voxel_size,
            model_axial_voxel_size=args.model_axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            model_lateral_voxel_size=args.model_lateral_voxel_size,
            wavelength=args.wavelength,
            scalar=args.scalar,
            threshold=args.threshold,
            plot=args.plot
        )

    elif args.func == 'predict_rois':
        experimental.predict_rois(
            model=args.model,
            img=args.input,
            peaks=args.peaks,
            dm_pattern=args.pattern,
            dm_state=args.state,
            prev=args.prev,
            psf_type=args.psf_type,
            axial_voxel_size=args.axial_voxel_size,
            model_axial_voxel_size=args.model_axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            model_lateral_voxel_size=args.model_lateral_voxel_size,
            wavelength=args.wavelength,
            scalar=args.scalar,
            threshold=args.threshold,
            plot=args.plot,
            window_size=args.window_size,
            num_rois=args.num_rois,
            min_intensity=args.min_intensity,
        )

    else:
        logger.error(f"Error")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
