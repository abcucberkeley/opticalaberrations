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
        "--skew_angle", default=32.45, type=float, help='skew angle'
    )
    deskew.add_argument(
        "--flipz", action='store_true',
        help='a toggle to flip Z axis'
    )

    decon = subparsers.add_parser("decon")
    decon.add_argument("input", type=Path, help="path to input .tif file")
    decon.add_argument("psf", type=Path, help="path to PSF .tif file")
    decon.add_argument(
        "--iters", default=10, type=Path,
        help="number of iterations for Richardson-Lucy deconvolution")
    decon.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )

    detect_rois = subparsers.add_parser("detect_rois")
    detect_rois.add_argument("input", type=Path, help="path to input .tif file")
    detect_rois.add_argument("--psf", default=None, type=Path, help="path to PSF .tif file")
    detect_rois.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    detect_rois.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )

    predict_sample = subparsers.add_parser("predict_sample")
    predict_sample.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_sample.add_argument("input", type=Path, help="path to input .tif file")
    predict_sample.add_argument("pattern", type=Path, help="path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)")
    predict_sample.add_argument(
        "--state", default=None, type=Path,
        help="optional path to current DM state .csv file (Default: `blank mirror`)"
    )
    predict_sample.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_sample.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        type=str, help='type of the desired PSF'
    )
    predict_sample.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_sample.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_sample.add_argument(
        "--model_lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_sample.add_argument(
        "--model_axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    predict_sample.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_sample.add_argument(
        "--scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    predict_sample.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_sample.add_argument(
        "--sign_threshold", default=.4, type=float,
        help='flip sign of modes above given threshold'
    )
    predict_sample.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_sample.add_argument(
        "--num_predictions", default=10, type=int,
        help="number of predictions per sample to estimate model's confidence"
    )
    predict_sample.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )

    predict_rois = subparsers.add_parser("predict_rois")
    predict_rois.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_rois.add_argument("input", type=Path, help="path to input .tif file")
    predict_rois.add_argument("peaks", type=Path, help="path to point detection results (.mat file)")

    predict_rois.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_rois.add_argument(
        "--window_size", default=64, type=int, help='size of the window to crop around each point of interest'
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
        "--num_rois", default=10, type=int,
        help='max number of detected points to use for estimating aberrations'
    )
    predict_rois.add_argument(
        "--min_intensity", default=200, type=int,
        help='minimum intensity desired for detecting peaks of interest'
    )
    predict_rois.add_argument(
        "--minimum_distance", default=.5, type=float,
        help='minimum distance to the nearest neighbor (microns)'
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
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_rois.add_argument(
        "--sign_threshold", default=.4, type=float,
        help='flip sign of modes above given threshold'
    )
    predict_rois.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_rois.add_argument(
        "--num_predictions", default=10, type=int,
        help="number of predictions per ROI to estimate model's confidence"
    )

    predict_tiles = subparsers.add_parser("predict_tiles")
    predict_tiles.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_tiles.add_argument("input", type=Path, help="path to input .tif file")

    predict_tiles.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_tiles.add_argument(
        "--window_size", default=64, type=int, help='size of the window to crop around each point of interest'
    )
    predict_tiles.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_tiles.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat', type=str,
        help='type of the desired PSF'
    )
    predict_tiles.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_tiles.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_tiles.add_argument(
        "--model_lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_tiles.add_argument(
        "--model_axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )
    predict_tiles.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_tiles.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_tiles.add_argument(
        "--sign_threshold", default=.4, type=float,
        help='flip sign of modes above given threshold'
    )
    predict_tiles.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_tiles.add_argument(
        "--num_predictions", default=10, type=int,
        help="number of predictions per tile to estimate model's confidence"
    )

    aggregate_predictions = subparsers.add_parser("aggregate_predictions")
    aggregate_predictions.add_argument("predictions", type=Path, help="path to csv file")
    aggregate_predictions.add_argument("input", type=Path, help="path to input .tif file")
    aggregate_predictions.add_argument("pattern", type=Path, help="path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)")

    aggregate_predictions.add_argument(
        "--state", default=None, type=Path,
        help="optional path to current DM state .csv file (Default: `blank mirror`)"
    )
    aggregate_predictions.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        type=str, help='type of the desired PSF'
    )
    aggregate_predictions.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    aggregate_predictions.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    aggregate_predictions.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    aggregate_predictions.add_argument(
        "--scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    aggregate_predictions.add_argument(
        "--prediction_threshold", default=.1, type=float,
        help='set predictions below threshold to zero (waves)'
    )
    aggregate_predictions.add_argument(
        "--majority_threshold", default=.5, type=float,
        help='majority rule to use to determine dominant modes among ROIs'
    )
    aggregate_predictions.add_argument(
        "--final_prediction", default='mean', type=str,
        help='rule to use to calculate final prediction [mean, median, min, max]'
    )
    aggregate_predictions.add_argument(
        "--min_percentile", default=10, type=int,
        help='minimum percentile to filter out outliers'
    )
    aggregate_predictions.add_argument(
        "--max_percentile", default=90, type=int,
        help='maximum percentile to filter out outliers'
    )
    aggregate_predictions.add_argument(
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
            skew_angle=args.skew_angle,
        )

    elif args.func == 'decon':
        experimental.decon(
            img=args.input,
            psf=args.psf,
            iters=args.iters,
            plot=args.plot,
        )

    elif args.func == 'detect_rois':
        experimental.detect_rois(
            img=args.input,
            psf=args.psf,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
        )

    elif args.func == 'predict_sample':
        experimental.predict_sample(
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
            prediction_threshold=args.prediction_threshold,
            sign_threshold=args.sign_threshold,
            num_predictions=args.num_predictions,
            plot=args.plot,
            batch_size=args.batch_size
        )

    elif args.func == 'predict_rois':
        experimental.predict_rois(
            model=args.model,
            img=args.input,
            peaks=args.peaks,
            prev=args.prev,
            psf_type=args.psf_type,
            axial_voxel_size=args.axial_voxel_size,
            model_axial_voxel_size=args.model_axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            model_lateral_voxel_size=args.model_lateral_voxel_size,
            wavelength=args.wavelength,
            window_size=args.window_size,
            num_predictions=args.num_predictions,
            num_rois=args.num_rois,
            min_intensity=args.min_intensity,
            prediction_threshold=args.prediction_threshold,
            sign_threshold=args.sign_threshold,
            minimum_distance=args.minimum_distance,
            plot=args.plot,
            batch_size=args.batch_size
        )
    elif args.func == 'predict_tiles':
        experimental.predict_tiles(
            model=args.model,
            img=args.input,
            prev=args.prev,
            psf_type=args.psf_type,
            prediction_threshold=args.prediction_threshold,
            sign_threshold=args.sign_threshold,
            axial_voxel_size=args.axial_voxel_size,
            model_axial_voxel_size=args.model_axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            model_lateral_voxel_size=args.model_lateral_voxel_size,
            num_predictions=args.num_predictions,
            wavelength=args.wavelength,
            window_size=args.window_size,
            plot=args.plot,
            batch_size=args.batch_size
        )
    elif args.func == 'aggregate_predictions':
        experimental.aggregate_predictions(
            data=args.input,
            model_pred=args.predictions,
            dm_pattern=args.pattern,
            dm_state=args.state,
            wavelength=args.wavelength,
            psf_type=args.psf_type,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            prediction_threshold=args.prediction_threshold,
            majority_threshold=args.majority_threshold,
            min_percentile=args.min_percentile,
            max_percentile=args.max_percentile,
            final_prediction=args.final_prediction,
            scalar=args.scalar,
            plot=args.plot,
        )
    else:
        logger.error(f"Error")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
