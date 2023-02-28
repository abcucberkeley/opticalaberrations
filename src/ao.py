import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import time
from pathlib import Path
import cli
import experimental
from preloaded import Preloadedmodelclass


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
        "--iters", default=10, type=int,
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
    predict_sample.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    predict_sample.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )
    predict_sample.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_sample.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_sample.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_sample.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_sample.add_argument(
        "--dm_damping_scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    predict_sample.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_sample.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_sample.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_sample.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_sample.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_sample.add_argument(
        "--num_predictions", default=1, type=int,
        help="number of predictions per sample to estimate model's confidence"
    )
    predict_sample.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_sample.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_sample.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )
    predict_sample.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )

    predict_rois = subparsers.add_parser("predict_rois")
    predict_rois.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_rois.add_argument("input", type=Path, help="path to input .tif file")
    predict_rois.add_argument("pois", type=Path, help="path to point detection results (.mat file)")

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
        "--psf_type", default='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat', type=Path,
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
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_rois.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_rois.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_rois.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_rois.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_rois.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_rois.add_argument(
        "--num_predictions", default=10, type=int,
        help="number of predictions per ROI to estimate model's confidence"
    )
    predict_rois.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_rois.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for modes you wish to ignore'
    )
    predict_rois.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )

    predict_tiles = subparsers.add_parser("predict_tiles")
    predict_tiles.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    predict_tiles.add_argument("input", type=Path, help="path to input .tif file")

    predict_tiles.add_argument(
        "--batch_size", default=100, type=int, help='maximum batch size for the model'
    )
    predict_tiles.add_argument(
        "--window_size", default=64, type=int, help='size of the window to crop each tile'
    )
    predict_tiles.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    predict_tiles.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat', type=Path,
        help='type of the desired PSF'
    )
    predict_tiles.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    predict_tiles.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    predict_tiles.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    predict_tiles.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    predict_tiles.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    predict_tiles.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    predict_tiles.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    predict_tiles.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    predict_tiles.add_argument(
        "--num_predictions", default=10, type=int,
        help="number of predictions per tile to estimate model's confidence"
    )
    predict_tiles.add_argument(
        "--estimate_sign_with_decon", action='store_true',
        help='a toggle for estimating signs of each Zernike mode via decon'
    )
    predict_tiles.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )
    predict_tiles.add_argument(
        "--ideal_empirical_psf", default=None, type=Path,
        help='path to an ideal empirical psf (Default: `None` ie. will be simulated automatically)'
    )

    aggregate_predictions = subparsers.add_parser("aggregate_predictions")
    aggregate_predictions.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    aggregate_predictions.add_argument("predictions", type=Path, help="path to csv file")
    aggregate_predictions.add_argument("dm_calibration", type=Path,
                                       help="path DM calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)")

    aggregate_predictions.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM current_dm .csv file (Default: `blank mirror`)"
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
        "--dm_damping_scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    aggregate_predictions.add_argument(
        "--prediction_threshold", default=.05, type=float,
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
    aggregate_predictions.add_argument(
        "--ignore_tile", action='append', default=None,
        help='IDs [e.g., "z0-y0-x0"] for tiles you wish to ignore'
    )

    eval_mode = subparsers.add_parser("eval_mode")
    eval_mode.add_argument("model_path", type=Path, help="path to pretrained tensorflow model (.h5)")
    eval_mode.add_argument("input_path", type=Path, help="path to input file (.tif)")
    eval_mode.add_argument("gt_path", type=Path, help="path to ground truth file (.csv)")
    eval_mode.add_argument("prediction_path", type=Path, help="path to model predictions (.csv)")
    eval_mode.add_argument("--prediction_postfix", type=str, default='sample_predictions_zernike_coefficients.csv')
    eval_mode.add_argument("--gt_postfix", type=str, default='ground_truth_zernike_coefficients.csv')

    eval_dataset = subparsers.add_parser("eval_dataset")
    eval_dataset.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    eval_dataset.add_argument("datadir", type=Path, help="path to dataset directory")
    eval_dataset.add_argument("--flat", default=None, type=Path, help="path to the flat DM acts file")
    eval_dataset.add_argument("--skip_eval_plots", action='store_true', help="skip generating the _ml_eval.svg files.")
    eval_dataset.add_argument("--precomputed", action='store_true')


    eval_dm = subparsers.add_parser("eval_dm")
    eval_dm.add_argument("datadir", type=Path, help="path to dataset directory")

    calibrate_dm = subparsers.add_parser("calibrate_dm")
    calibrate_dm.add_argument("datadir", type=Path, help="path to DM eval directory")
    calibrate_dm.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )


    phase_retrieval = subparsers.add_parser("phase_retrieval")
    phase_retrieval.add_argument("input", type=Path, help="path to input .tif file")
    phase_retrieval.add_argument(
        "dm_calibration", type=Path,
        help="path DM dm_calibration mapping matrix (eg. Zernike_Korra_Bax273.csv)"
    )
    phase_retrieval.add_argument(
        "--num_modes", type=int, default=15,
        help="number of zernike modes to predict"
    )
    phase_retrieval.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )
    phase_retrieval.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )
    phase_retrieval.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    phase_retrieval.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    phase_retrieval.add_argument(
        "--dm_damping_scalar", default=.75, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )
    phase_retrieval.add_argument(
        "--prediction_threshold", default=0., type=float,
        help='set predictions below threshold to zero (waves)'
    )
    phase_retrieval.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    phase_retrieval.add_argument(
        "--num_iterations", default=150, type=int,
        help="max number of iterations"
    )
    phase_retrieval.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )

    phase_retrieval.add_argument(
        "--use_pyotf_zernikes", action='store_true',
        help='a toggle to use pyOTF zernike definitions'
    )

    return parser.parse_args(args)


def main(args=None, preloaded: Preloadedmodelclass = None):

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
            dm_calibration=args.dm_calibration,
            dm_state=args.current_dm,
            prev=args.prev,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            wavelength=args.wavelength,
            dm_damping_scalar=args.dm_damping_scalar,
            freq_strength_threshold=args.freq_strength_threshold,
            prediction_threshold=args.prediction_threshold,
            sign_threshold=args.sign_threshold,
            num_predictions=args.num_predictions,
            plot=args.plot,
            plot_rotations=args.plot_rotations,
            batch_size=args.batch_size,
            estimate_sign_with_decon=args.estimate_sign_with_decon,
            ignore_modes=args.ignore_mode,
            ideal_empirical_psf=args.ideal_empirical_psf,
            preloaded=preloaded
        )

    elif args.func == 'predict_rois':
        experimental.predict_rois(
            model=args.model,
            img=args.input,
            pois=args.pois,
            prev=args.prev,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            wavelength=args.wavelength,
            window_size=args.window_size,
            num_predictions=args.num_predictions,
            num_rois=args.num_rois,
            min_intensity=args.min_intensity,
            freq_strength_threshold=args.freq_strength_threshold,
            prediction_threshold=args.prediction_threshold,
            sign_threshold=args.sign_threshold,
            minimum_distance=args.minimum_distance,
            plot=args.plot,
            plot_rotations=args.plot_rotations,
            batch_size=args.batch_size,
            estimate_sign_with_decon=args.estimate_sign_with_decon,
            ignore_modes=args.ignore_mode,
            ideal_empirical_psf=args.ideal_empirical_psf,
            preloaded=preloaded
        )
    elif args.func == 'predict_tiles':
        experimental.predict_tiles(
            model=args.model,
            img=args.input,
            prev=args.prev,
            freq_strength_threshold=args.freq_strength_threshold,
            prediction_threshold=args.prediction_threshold,
            sign_threshold=args.sign_threshold,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            num_predictions=args.num_predictions,
            wavelength=args.wavelength,
            window_size=args.window_size,
            plot=args.plot,
            plot_rotations=args.plot_rotations,
            batch_size=args.batch_size,
            estimate_sign_with_decon=args.estimate_sign_with_decon,
            ignore_modes=args.ignore_mode,
            ideal_empirical_psf=args.ideal_empirical_psf,
            preloaded=preloaded
        )
    elif args.func == 'aggregate_predictions':
        experimental.aggregate_predictions(
            model=args.model,
            model_pred=args.predictions,
            dm_calibration=args.dm_calibration,
            dm_state=args.current_dm,
            wavelength=args.wavelength,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            prediction_threshold=args.prediction_threshold,
            majority_threshold=args.majority_threshold,
            min_percentile=args.min_percentile,
            max_percentile=args.max_percentile,
            final_prediction=args.final_prediction,
            ignore_tile=args.ignore_tile,
            dm_damping_scalar=args.dm_damping_scalar,
            plot=args.plot,
            preloaded=preloaded
        )
    elif args.func == 'eval_dataset':
        experimental.eval_dataset(
            model=args.model,
            datadir=args.datadir,
            flat=args.flat,
            plot_evals=not args.skip_eval_plots,
            precomputed=args.precomputed,
        )
    elif args.func == 'eval_mode':
        experimental.eval_mode(
            model_path=args.model_path,
            input_path=args.input_path,
            prediction_path=args.prediction_path,
            gt_path=args.gt_path,
            postfix=args.prediction_postfix,
            gt_postfix=args.gt_postfix,
        )
    elif args.func == 'eval_dm':
        experimental.eval_dm(
            datadir=args.datadir,
        )
    elif args.func == 'calibrate_dm':
        experimental.calibrate_dm(
            datadir=args.datadir,
            dm_calibration=args.dm_calibration,
        )
    elif args.func == 'phase_retrieval':
        experimental.phase_retrieval(
            img=args.input,
            num_modes=args.num_modes,
            dm_calibration=args.dm_calibration,
            dm_state=args.current_dm,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            wavelength=args.wavelength,
            dm_damping_scalar=args.dm_damping_scalar,
            prediction_threshold=args.prediction_threshold,
            num_iterations=args.num_iterations,
            plot=args.plot,
            ignore_modes=args.ignore_mode,
            use_pyotf_zernikes=args.use_pyotf_zernikes,
        )
    else:
        logger.error(f"Error")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
