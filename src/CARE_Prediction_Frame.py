from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE

# predict for inputFullpath with model in given modelPath, the result writes to outputFullpath
# set negative values as 0
 

def CARE_Prediction_Frame(inputFullpath, outputFullpath, modelPath, subtractVal=0., n_tiles=(4,10,10), axes='ZYX', Save16bit=False):
    # if os.path.isfile(outputFullpath):
    #     print(f"output file exists, skip it! :  {outputFullpath.resolve()}")
    #     return

    model = CARE(config=None, name=modelPath.name, basedir=modelPath.parent)
    print(f"CARE model loaded : {modelPath}")
    print(inputFullpath)
    # fnbase = os.path.basename(inputFullpath)
    print (f"output file does not exist... Processing using {n_tiles=}")
    x = imread(inputFullpath)
    x = x.astype('float32') - subtractVal
    
    restored = model.predict(x, axes, n_tiles=n_tiles)
    # Path(srt).mkdir(exist_ok=True)
    restored[restored < 0.0] = 0.0
    if Save16bit:
        save_tiff_imagej_compatible(outputFullpath, restored.astype('uint16'), axes, compression='zlib')
    else:
        save_tiff_imagej_compatible(outputFullpath, restored.astype('float32'), axes, compression='zlib')

    print(f"Done! Saved to {outputFullpath}")


def main(inputFullpath=None, outputFullpath=None, modelPath=None, basedir=None, subtractVal=0., n_tiles=(4,10,10), axes='ZYX', Save16bit=False):
    
    if len(sys.argv) > 1:
        inputFullpath = sys.argv[1]
        outputFullpath = sys.argv[2]
        modelPath = sys.argv[3]
        basedir = sys.argv[4]
        Save16bit = sys.argv[5].lower() == 'true'
        subtractVal = float(sys.argv[6])
        n_tiles = [float(nt) for nt in sys.argv[7].split(',')]
        n_tiles = tuple(n_tiles)

    print(sys.argv)
    print(inputFullpath)
    print(outputFullpath)
    print(modelPath)
    print(basedir)
    print(Save16bit)
    print(subtractVal)
    print(n_tiles)
    CARE_Prediction_Frame(inputFullpath, outputFullpath, modelPath, basedir, subtractVal, n_tiles, axes, Save16bit)


if __name__ == "__main__":
    main()




