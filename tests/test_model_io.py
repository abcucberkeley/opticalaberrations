import sys
import os
import numpy as np
from pathlib import Path
import platform
from tensorflow import config as tfc

sys.path.insert(1, r'C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src')
# needed to get to the src folder of the project so that everything below will import from src
   
import backend

def test_model_io():
    os.chdir(r'C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src')    # need working directory to be src because .mat location (via relative path) is stored in model .h5 that we will load
    print(f"Here is my cwd:{os.getcwd()}")         # opticalaberrations\src
    modelpath   = r'C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\pretrained_models\z60_modes\lattice_yumb\x108-y108-z200'
    modelfile   = 'opticaltransformer.h5'

    if platform.system() == "Windows":
        import win32api  # pip install pywin32
        modelpath = Path(win32api.GetShortPathName(modelpath).lower())   # shorten name to get rid of spaces.
    else:
        modelpath = Path(modelpath)

    modelpath = Path(modelpath, modelfile)


    print(f"my model is here: {modelpath}")
    print(f"Has extension={modelpath.suffix}")
    physical_devices = tfc.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tfc.experimental.set_memory_growth(gpu_instance, True)

    modelpsfgen = backend.load_metadata(modelpath)
    print("   ****metadata loaded.***   ")
    model = backend.load(modelpath, mosaic=True)
    print("   ****model loaded.***   ")
    assert True
    return f"Model was loaded from: {modelpath}"
