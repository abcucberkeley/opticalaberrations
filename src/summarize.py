from pathlib import Path
import os
import numpy as np
from tifffile import TiffFile, imwrite, imread, TiffWriter

def concat_U16_tiffs(source_files=list([]), destination:Path=None, next_channel_files=None, patterns_to_drop=list([])):
    """
    Make a hyperstack with the source_files listed.  Will append the "below_files" to the bottom of the images.
    Args:
        source_files: Generator or List of files
        destination: Path of output .tiff
        below_files: Generator or List of files to place at the bottom of the before_images
        patterns_to_drop: source_files will be dropped if the Path contains any string in this list of strings

    Returns:

    """
    source_files = [x for x in source_files if all(y not in str(x) for y in patterns_to_drop)]
    source_files.sort(key=lambda x: os.path.getmtime(x))    # sort by modified time
    if next_channel_files is not None:
        next_channel_files = [x for x in next_channel_files if all(y not in str(x) for y in patterns_to_drop)]
        next_channel_files.sort(key=lambda x: os.path.getmtime(x))  # sort by modified time
        c_size = 2
    else:
        c_size = 1

    t_size = len(source_files)
    if t_size == 0:
        print(f'Warning: Did not find any source files to make {destination}\n')
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
                f"Concatenating {i + 1:2} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")
        if c_size == 2:
            with TiffFile(next_channel_files[i]) as tif:
                hyperstack[i, :, 1] = tif.asarray().astype(data_type)
                print(
                    f"Concatenating {i + 1:2} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    image_labels = [x.stem[:40] for x in source_files]
    image_labels = list(np.repeat(image_labels, z_size))  # every slice needs a label

    imwrite(destination,
            hyperstack,
            dtype=data_type,
            imagej=True,
            metadata={
                'axes': axes_string,
                'Labels': image_labels,
            },
            )
    print(f"Saved:\n{destination.resolve()}\n")


folder = r'U:\Data\TestsForThayer\20230908_fish\exp10_ap2fish1_eye_jh\rotated'
folder = Path(folder)
before_files = folder.glob('*CamA*stack0000_*00??t.tif')
optimized_files = folder.glob('*CamA*stack0000_*00??*optimized.tif')
vol_used_files = folder.glob('*CamA*stack0000_*00??*volume_used.tif')
patterns_to_drop = list(['after_three'])


destination = Path(f"{folder}\\_summary\\{folder.parts[-2]}_before_hyperstack.tif")
concat_U16_tiffs(source_files=before_files, destination=destination, patterns_to_drop=patterns_to_drop)

before_files = folder.glob('*CamA*stack0000_*00??t.tif')
destination = Path(f"{folder}\\_summary\\{folder.parts[-2]}_before_vs_optimized_hyperstack.tif")
concat_U16_tiffs(source_files=before_files, destination=destination, patterns_to_drop=patterns_to_drop, next_channel_files=optimized_files)

destination = Path(f"{folder}\\_summary\\{folder.parts[-2]}_optimized_hyperstack.tif")
concat_U16_tiffs(source_files=optimized_files, destination=destination, patterns_to_drop=patterns_to_drop)

destination = Path(f"{folder}\\_summary\\{folder.parts[-2]}_volume_used_hyperstack.tif")
concat_U16_tiffs(source_files=vol_used_files, destination=destination, patterns_to_drop=patterns_to_drop)

consensus_clusters = folder.glob('*_combined_tiles_predictions_consensus_clusters.tif')
consensus_clusters_wavefronts = folder.glob('*_combined_tiles_predictions_consensus_clusters_wavefronts.tif')
consensus_clusters_psfs = folder.glob('*_combined_tiles_predictions_consensus_clusters_psfs.tif')
patterns_to_drop = list(['after_three', 'pythons_great'])
destination = Path(f"{folder}\\_summary\\{folder.parts[-2]}_consensus_map.tif")

consensus_clusters = [x for x in consensus_clusters if all(y not in str(x) for y in patterns_to_drop)]
consensus_clusters.sort(key=lambda x: os.path.getmtime(x))

consensus_clusters_wavefronts = [x for x in consensus_clusters_wavefronts if all(y not in str(x) for y in patterns_to_drop)]
consensus_clusters_wavefronts.sort(key=lambda x: os.path.getmtime(x))

consensus_clusters_psfs = [x for x in consensus_clusters_psfs if all(y not in str(x) for y in patterns_to_drop)]
consensus_clusters_psfs.sort(key=lambda x: os.path.getmtime(x))

t_size = len(consensus_clusters)
sample = TiffFile(consensus_clusters[0])
z_size = len(sample.pages)  # number of pages in the file
page = sample.pages[0]  # get shape and dtype of image in first page
(y_size, x_size, c_size) = page.shape   # c_size = 3 for color image

hyperstack = np.zeros([t_size, z_size, y_size*2, x_size, c_size ]).astype(np.ubyte) # vertically combine these two stacks.
hyperstack = np.squeeze(hyperstack)

for i in range(len(consensus_clusters)):

    with TiffFile(consensus_clusters[i]) as tif:
        hyperstack[i,:,:y_size] = tif.asarray()
        print(f"Concatenating {i+1} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")

    with TiffFile(consensus_clusters_wavefronts[i]) as tif:
        hyperstack[i,:,y_size:] = tif.asarray()
        print(f"Concatenating {i+1} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")

    # with TiffFile(consensus_clusters_psfs[i]) as tif:
    #     hyperstack[i,:,y_size*2:] = np.repeat(tif.asarray(), z_size/2, axis=0) # since this stack only has 1 slice per z slab, we repeat to fill out.
    #     print(f"Concatenating {i+1} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")


destination.parent.mkdir(parents=True, exist_ok=True)
image_labels = [x.stem[:40] + '...' + x.stem[-20:] for x in consensus_clusters]
image_labels = list(np.repeat(image_labels, z_size)) # every slice needs a label
imwrite(destination,
        hyperstack.astype(np.ubyte),
        photometric='rgb',
        imagej=True,
        metadata={
            'axes': 'TZYXS',
            'Labels': image_labels,
            },
        )

print(f"\nSaved:\n{destination.resolve()}")
