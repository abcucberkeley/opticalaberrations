from pathlib import Path
import os
import numpy as np
from tifffile import TiffFile, imwrite, imread, TiffWriter

folder = r'U:\Data\TestsForThayer\20230825_ap2\exp3_newfisth\rotated'
folder = Path(folder)
before_files = folder.glob('*CamA*stack0000_*00??t.tif')
patterns_to_drop = list(['after_three', 'pythons_great'])
destination = Path(f"{folder}\\_summary\\before_hyperstack.tif")

before_files = [x for x in before_files if all(y not in str(x) for y in patterns_to_drop)]
before_files.sort(key=lambda x: os.path.getmtime(x))
# print(before_files)

t_size = len(before_files)
sample = TiffFile(before_files[0])
z_size = len(sample.pages)  # number of pages in the file
page = sample.pages[0]  # get shape and dtype of image in first page
y_size, x_size = page.shape

hyperstack = np.zeros([t_size, z_size, y_size, x_size ]).astype(np.float32)

for i in range(len(before_files)):
    with TiffFile(before_files[i]) as tif:
        hyperstack[i] = tif.asarray()
        print(f"Concatenating {i+1} out of {t_size} ({len(sample.pages)} x {sample.pages[0].shape[0]} x {sample.pages[0].shape[1]}) {tif.filename}")


destination.parent.mkdir(parents=True, exist_ok=True)
image_labels = [x.stem[:40] for x in before_files]
image_labels = list(np.repeat(image_labels, z_size)) # every slice needs a label
imwrite(destination,
        hyperstack,
        dtype=np.float32,
        imagej=True,
        metadata={
            'axes': 'TZYX',
            'Labels': image_labels,
            },
        )

print(f"\nSaved:\n{destination.resolve()}")
