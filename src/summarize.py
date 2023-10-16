from pathlib import Path
import os
import numpy as np
from tifffile import TiffFile, imwrite, imread, TiffWriter


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
    image_labels = [x.stem[:40] for x in source_files]
    image_labels = list(np.repeat(image_labels, z_size))  # every slice needs a label

    imwrite(dst,
            hyperstack,
            dtype=data_type,
            imagej=True,
            metadata={
                'axes': axes_string,
                'Labels': image_labels,
            },
            )
    print(f"Saved:\n{dst.resolve()}\n")


folder = Path(r'U:\Data\TestsForThayer\20231013_fish\exp1_notochord\rotated')
cam_A = 'CamA'
cam_B = 'CamB'

before_files = list(folder.glob(f'*{cam_A}*stack0000_*00??t.tif'))
before_files_b = list(folder.glob(f'*{cam_B}*stack0000_*00??t.tif'))
optimized_files = list(folder.glob(f'*{cam_A}*stack0000_*00??*optimized.tif'))
optimized_files_b = list(folder.glob(f'*{cam_B}*stack0000_*00??*optimized.tif'))
vol_used_files = list(folder.glob(f'*{cam_A}*stack0000_*00??*volume_used.tif'))
patterns_to_drop = list(['after_three'])


dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_before_hyperstack_{cam_A}.tif")
concat_U16_tiffs(source_files=before_files, dst=dst, drop_patterns=patterns_to_drop)
dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_before_hyperstack_{cam_B}.tif")
concat_U16_tiffs(source_files=before_files_b, dst=dst, drop_patterns=patterns_to_drop)

dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_before_vs_optimized_hyperstack_{cam_A}.tif")
concat_U16_tiffs(source_files=before_files, dst=dst, drop_patterns=patterns_to_drop, ch_two=optimized_files)
dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_before_vs_optimized_hyperstack_{cam_B}.tif")
concat_U16_tiffs(source_files=before_files_b, dst=dst, drop_patterns=patterns_to_drop, ch_two=optimized_files_b)

dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_optimized_hyperstack_{cam_A}.tif")
concat_U16_tiffs(source_files=optimized_files, dst=dst, drop_patterns=patterns_to_drop)
dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_optimized_hyperstack_{cam_B}.tif")
concat_U16_tiffs(source_files=optimized_files_b, dst=dst, drop_patterns=patterns_to_drop)

dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_volume_used_hyperstack.tif")
concat_U16_tiffs(source_files=vol_used_files, dst=dst, drop_patterns=patterns_to_drop)


# make consensus_map (aka wavefronts over time)
dst = Path(f"{folder}\\_summary\\{folder.parts[-2]}_consensus_map.tif")
consensus_clusters = folder.glob('*_combined_tiles_predictions_consensus_clusters.tif')
consensus_clusters_wavefronts = folder.glob('*_combined_tiles_predictions_consensus_clusters_wavefronts.tif')
consensus_clusters_psfs = folder.glob('*_combined_tiles_predictions_consensus_clusters_psfs.tif')

# filter files via "patterns_to_drop", then sort by modified time.
consensus_clusters = [x for x in consensus_clusters if all(y not in str(x) for y in patterns_to_drop)]
consensus_clusters.sort(key=lambda x: os.path.getmtime(x))

consensus_clusters_wavefronts = [x for x in consensus_clusters_wavefronts if all(y not in str(x) for y in patterns_to_drop)]
consensus_clusters_wavefronts.sort(key=lambda x: os.path.getmtime(x))

consensus_clusters_psfs = [x for x in consensus_clusters_psfs if all(y not in str(x) for y in patterns_to_drop)]
consensus_clusters_psfs.sort(key=lambda x: os.path.getmtime(x))

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
imwrite(dst,
        hyperstack.astype(np.ubyte),
        photometric='rgb',
        imagej=True,
        metadata={
            'axes': 'TZYXS',
            'Labels': image_labels,
        },
        )

print(f"\nSaved:\n{dst.resolve()}")