from pathlib import Path
from tensorflow import config as tfc
from backend import load, load_metadata


class Preloadedmodelclass:
    """ A class that LabVIEW can use to keep the model in memory to prevent reloading it each time.
    """
    def __init__(self, modelpath: Path, **kwargs):
        self.modelpath = Path(modelpath)

        physical_devices = tfc.list_physical_devices('GPU')
        for gpu_instance in physical_devices:
            tfc.experimental.set_memory_growth(gpu_instance, True)

        self.modelpsfgen = load_metadata(self.modelpath, **kwargs)
        self.model = load(self.modelpath, mosaic=True)

