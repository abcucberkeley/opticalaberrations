import logging
import time
from pathlib import Path
import os


import cli
import imghdr


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    parser.add_argument("input", type=Path, help="path to input .tif file")
    parser.add_argument("pattern", type=Path, help="path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)")

    parser.add_argument(
        "--state", default=None, type=Path,
        help="optional path to current DM state .csv file (Default: `blank mirror`)"
    )

    parser.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )

    parser.add_argument(
        "--psf_type", default='../examples/lattice/lattice_PSF_simulation.mat', type=str, help='type of the desired PSF'
    )

    parser.add_argument(
        "--lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--axial_voxel_size", default=.268, type=float, help='axial voxel size in microns for Z'
    )

    parser.add_argument(
        "--model_lateral_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--model_axial_voxel_size", default=.268, type=float, help='axial voxel size in microns for Z'
    )

    parser.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )

    parser.add_argument(
        "--scalar", default=1.1, type=float,
        help='scale DM actuators by an arbitrary multiplier'
    )

    parser.add_argument(
        "--threshold", default=1e-2, type=float,
        help='set predictions below threshold to zero (microns)'
    )

    parser.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )

    parser.add_argument(
        "--verbose", action='store_true',
        help='a toggle for a progress bar'
    )

    return parser.parse_args(args)


def main(args=None):

    timeit = time.time()
    args = parse_args(args)
    logging.info(args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger('')

    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        import experimental

    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import experimental

    if imghdr.what(args.input) == 'tiff':
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
            verbose=args.verbose,
            plot=args.plot
        )

    else:
        logger.error(f"Error: Input file format is not tiff, instead is: {imghdr.what(args.input)} File: {args.input}")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


def make_a_logger(myinput: Path):
    """
        Creates a logger that will output to the console and to a log.txt file
        in the same folder as the file specified by 'myinput'.
    """

    # Create a custom logger
    # get root logger.  FYI, this "root logger" is called from experimental.py and other places via globals
    logger = logging.getLogger('')

    # don't duplicate handlers if we already have them
    if logger.hasHandlers() == False:
        log_filepath = Path(myinput.parent, 'log.txt')
        print(f'Logging output to: {log_filepath}')

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_filepath, mode='w')

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(c_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.setLevel(logging.INFO)  # have to put the logging level here not at the handler level. sigh.

    return logger


def main_function(
    mymodel: str,
    myinput: str,
    mypattern: str,
    myprev: str,
    mystate: str,
    psf_type: str,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    model_axial_voxel_size: float,
    model_lateral_voxel_size: float,
    mywavelength: float,
    myscalar: float = 1.,
    mythreshold: float = 0.,
    verbose: bool = True,
    plot: bool = True
):
    """ Replicates main but using parameters so that we can call it from LabVIEW. """
    try:

        timestart = time.time()
               
        # python truly hates spaces in names. Can get short name running a batch file with @ECHO OFF echo %~s1
        mymodel = Path(mymodel)
        myinput = Path(myinput)
        mypattern = Path(mypattern)
        mystate = Path(mystate)
        myprev = Path(myprev)

        logger = make_a_logger(myinput)

        try:
            import matplotlib
            matplotlib.use('TkAgg')

        except ImportError:
            logger.error('The default Qt backend does not work with LabVIEW. Please install Tkinter!')

        try:
            if verbose:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
                import experimental

            else:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                import experimental

        except ImportError:
            return logger.error("Error: Please install tensorflow first!")

        if imghdr.what(myinput) == 'tiff':
            logger.info("Yes, input is a tiff file.")
            experimental.predict(
                model=mymodel,
                img=myinput,
                dm_pattern=mypattern,
                dm_state=mystate,
                prev=myprev,
                psf_type=psf_type,
                axial_voxel_size=axial_voxel_size,
                model_axial_voxel_size=model_axial_voxel_size,
                lateral_voxel_size=lateral_voxel_size,
                model_lateral_voxel_size=model_lateral_voxel_size,
                wavelength=mywavelength,
                scalar=myscalar,
                threshold=mythreshold,
                verbose=verbose,
                plot=plot,
            )
        else:
            logger.error(f"Error: Input file format is not tiff, instead is: {imghdr.what(myinput)} File: {myinput}")

        logger.info(f"Total time elapsed: {time.time() - timestart:.2f} sec.")

    except BaseException as err:
        logging.exception("Exception occurred")


if __name__ == "__main__":
    main()
