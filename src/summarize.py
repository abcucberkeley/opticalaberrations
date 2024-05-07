from pathlib import Path
import os
import sys
import cli
import subprocess
import pandas as pd
import numpy as np
from tifffile import TiffFile, imwrite, imread, TiffWriter

from slurm_utils import paths_to_clusterfs

def concat_U16_tiffs(source_files=list([]), dst: Path = None, ch_two=None, drop_patterns=list([])):
    """
    Make a hyperstack with the source_files listed.  Will append the "below_files" to the bottom of the images.
    Args:        
        source_files: Generator or List of files
        dst: Path of output .tiff
        ch_two: Generator or List of files to place in the 2nd channel
        drop_patterns: source_files will be dropped if the Path contains any string in this list of strings

    Returns:

    """
    source_files = [x for x in source_files if all(y not in str(x) for y in drop_patterns)]
    source_files.sort(key=lambda x: os.path.getmtime(x))  # sort by modified time
    if ch_two is not None:
        ch_two = [x for x in ch_two if all(y not in str(x) for y in drop_patterns)]
        ch_two.sort(key=lambda x: os.path.getmtime(x))  # sort by modified time
        if len(ch_two) > 0:
            c_size = 2
            if len(ch_two) != len(source_files):
                print(f'Warning: Lengths of Source file list ({len(source_files)}) '
                      f'and second tiff channel ({len(ch_two)}) do not match.\n')
                return
        else:
            c_size =1

    else:
        c_size = 1

    t_size = len(source_files)
    if t_size == 0:
        print(f'Warning: Did not find any source files to make {dst}\n')
        return

    sample = TiffFile(source_files[0])
    z_size = len(sample.pages)  # number of pages in the file
    page = sample.pages[0]  # get shape and dtype of image in first page
    y_size, x_size = page.shape
    data_type = np.uint16
    axes_string = 'TZCYX'
    hyperstack = np.zeros([t_size, z_size, c_size, y_size, x_size]).astype(data_type)

    for i in range(len(source_files)):
        with TiffFile(source_files[i]) as tif:
            hyperstack[i, :, 0] = tif.asarray().astype(data_type)
            print(
                f"Concatenating {i + 1:2} out of {t_size} "
                f"({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")
        if c_size == 2:
            with TiffFile(ch_two[i]) as tif:
                hyperstack[i, :, 1] = tif.asarray().astype(data_type)
                print(
                    f"Concatenating {i + 1:2} out of {t_size} "
                    f"({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt' and input is not None:
        subprocess.run(f"chmod -R a+wrx {dst.parent}", shell=True)

    image_labels = [x.stem[:40] for x in source_files]
    image_labels = list(np.repeat(image_labels, z_size))  # every slice needs a label

    imwrite(
        dst,
        hyperstack,
        dtype=data_type,
        imagej=True,
        metadata={
            'axes': axes_string,
            'Labels': image_labels,
        },
        compression='deflate'
    )
    print(f"Saved:\n{dst.resolve()}")

    # Make MIPs
    dst =  Path(f"{dst.with_suffix('')}_mip.tif")
    imwrite(
        dst,
        np.max(hyperstack, axis=1),
        dtype=data_type,
        imagej=True,
        # metadata={
        #     'axes': axes_string,
        #     'Labels': image_labels,
        # },
        compression='deflate'
    )

    print(f"{dst.resolve()}\n")


def parse_args(args):
    parser = cli.argparser()
    parser.add_argument("input", type=Path, help=r"Path to input folder, for example:  U:\Data\TestsForThayer\20240123_cells\exp8-R2462\rotated")
    parser.add_argument(
        "--denoised", action='store_true',
        help='a toggle to run summarize on denoised data'
    )

    return parser.parse_known_args(args)

def main(args=None):
    command_flags = sys.argv[1:] if args is None else args
    args, unknown = parse_args(args)
    pd.options.display.width = 200
    pd.options.display.max_columns = 20

    folder = args.input
    if os.name != 'nt':
        folder = paths_to_clusterfs(folder, None)

    if os.name != 'nt' and not Path('/clusterfs').exists():
        mount_clusterfs = (r"sudo mkdir /clusterfs && sudo chmod a+wrx /clusterfs/ && "  # make empty directory
                           r"sudo chown 1000:1000 -R /sshkey/ && "  # make /sshkeys (was mounted from host) avail to user 1000
                           r"sshfs thayeralshaabi@master.abc.berkeley.edu:/clusterfs /clusterfs -oIdentityFile=/sshkey/id_rsa -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null ")  # sshfs mount without user input
        subprocess.run(mount_clusterfs, shell=True)
        subprocess.run('ls /clusterfs', shell=True)

    cam_A = 'CamA'
    cam_B = 'CamB'
    denoise_suffix = '_denoised'

    before_files = list(folder.glob(f'*{cam_A}*stack0000_*00??t{denoise_suffix}.tif'))
    if len(before_files) == 0:
        denoise_suffix = ''

    print(f'Looking for  *{cam_A}*stack0000_*00??t{denoise_suffix}.tif   in  {folder.resolve()}  where {len(list(folder.glob("*.tif")))} .tif files are found. '  )
    before_files = list(folder.glob(f'*{cam_A}*stack0000_*00??t{denoise_suffix}.tif'))
    before_files_b = list(folder.glob(f'*{cam_B}*stack0000_*00??t{denoise_suffix}.tif'))
    optimized_files = list(folder.glob(f'*{cam_A}*stack0000_*00??*{denoise_suffix}_optimized.tif'))
    optimized_files_b = list(folder.glob(f'*{cam_B}*stack0000_*00??*{denoise_suffix}_optimized.tif'))
    vol_used_files = list(folder.glob(f'*{cam_A}*stack0000_*00??*{denoise_suffix}_combined_volume_used.tif'))
    patterns_to_drop = list(['after_three'])

    if len(before_files) == 0:
        raise Exception("No files were found")

    dst = Path(f"{folder}/_summary/{folder.parts[-2]}_before_hyperstack_{cam_A}{denoise_suffix}.tif")
    concat_U16_tiffs(source_files=before_files, dst=dst, drop_patterns=patterns_to_drop)
    dst = Path(f"{folder}/_summary/{folder.parts[-2]}_before_hyperstack_{cam_B}{denoise_suffix}.tif")
    concat_U16_tiffs(source_files=before_files_b, dst=dst, drop_patterns=patterns_to_drop)

    if len(optimized_files) > 0:
        dst = Path(f"{folder}/_summary/{folder.parts[-2]}_before_vs_optimized_hyperstack_{cam_A}{denoise_suffix}.tif")
        concat_U16_tiffs(source_files=before_files, dst=dst, drop_patterns=patterns_to_drop, ch_two=optimized_files)
    if len(optimized_files_b) > 0:
        dst = Path(f"{folder}/_summary/{folder.parts[-2]}_before_vs_optimized_hyperstack_{cam_B}{denoise_suffix}.tif")
        concat_U16_tiffs(source_files=before_files_b, dst=dst, drop_patterns=patterns_to_drop, ch_two=optimized_files_b)

    if len(optimized_files) > 0:
        dst = Path(f"{folder}/_summary/{folder.parts[-2]}_optimized_hyperstack_{cam_A}{denoise_suffix}.tif")
        concat_U16_tiffs(source_files=optimized_files, dst=dst, drop_patterns=patterns_to_drop)
    if len(optimized_files_b) > 0:
        dst = Path(f"{folder}/_summary/{folder.parts[-2]}_optimized_hyperstack_{cam_B}{denoise_suffix}.tif")
        concat_U16_tiffs(source_files=optimized_files_b, dst=dst, drop_patterns=patterns_to_drop)

    dst = Path(f"{folder}/_summary/{folder.parts[-2]}_volume_used_hyperstack{denoise_suffix}.tif")
    concat_U16_tiffs(source_files=vol_used_files, dst=dst, drop_patterns=patterns_to_drop)

    # make consensus_map (aka wavefronts over time)
    dst = Path(f"{folder}/_summary/{folder.parts[-2]}_consensus_map{denoise_suffix}.tif")
    consensus_clusters = folder.glob(f'*{denoise_suffix}_combined_tiles_predictions_consensus_clusters.tif')
    consensus_clusters_wavefronts = folder.glob(f'*{denoise_suffix}_combined_tiles_predictions_consensus_clusters_wavefronts.tif')
    consensus_clusters_psfs = folder.glob(f'*{denoise_suffix}_combined_tiles_predictions_consensus_clusters_psfs.tif')

    # filter files via "patterns_to_drop", then sort by modified time.
    consensus_clusters = [x for x in consensus_clusters if all(y not in str(x) for y in patterns_to_drop)]
    consensus_clusters.sort(key=lambda x: os.path.getmtime(x))

    consensus_clusters_wavefronts = [x for x in consensus_clusters_wavefronts if all(y not in str(x) for y in patterns_to_drop)]
    consensus_clusters_wavefronts.sort(key=lambda x: os.path.getmtime(x))

    consensus_clusters_psfs = [x for x in consensus_clusters_psfs if all(y not in str(x) for y in patterns_to_drop)]
    consensus_clusters_psfs.sort(key=lambda x: os.path.getmtime(x))

    if len(consensus_clusters) > 0:
        t_size = len(consensus_clusters)    # number of time points
        sample = TiffFile(consensus_clusters[0])
        z_size = len(sample.pages)          # number of pages in the file
        page = sample.pages[0]              # get shape
        (y_size, x_size, c_size) = page.shape  # c_size = 3 for color image

        # vertically combine "consensus_clusters" and "psfs".
        hyperstack = np.zeros(shape=[t_size, z_size, y_size * 2, x_size, c_size], dtype=np.ubyte)
        hyperstack = np.squeeze(hyperstack)

        for i in range(len(consensus_clusters)):
            with TiffFile(consensus_clusters[i]) as tif:
                print(
                    f"Concatenating {i + 1} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")
                hyperstack[i, :, :y_size] = tif.asarray()   # place into top of image

            with TiffFile(consensus_clusters_psfs[i]) as tif:
                # since this stack only has 1 slice per z slab, we repeat to fill out.
                hyperstack[i, :, y_size:] = np.repeat(tif.asarray(), z_size//len(tif.pages), axis=0)    # place into bottom of image.
                print(f"Concatenating {i+1} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        image_labels = [x.stem[:40] + '...' + x.stem[-20:] for x in consensus_clusters]
        image_labels = list(np.repeat(image_labels, z_size))  # every slice needs a label
        imwrite(
            dst,
            hyperstack.astype(np.ubyte),
            photometric='rgb',
            imagej=True,
            metadata={
                'axes': 'TZYXS',
                'Labels': image_labels,
            },
            compression='deflate'
        )

        print(f"\nSaved:\n{dst.resolve()}")

    if os.name != 'nt':
        print(f"Updating file permissions to {dst.parent}")
        subprocess.run(f"find {str(dst.parent.resolve())}" + r" -user $USER -exec chmod a+wrx {} +",  shell=True)
        print(f"Updating file permissions complete.")

if __name__ == "__main__":
    main()