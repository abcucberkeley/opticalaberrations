import experimental
from pathlib import Path

class Preloadedmodelclass:
    """ A class that LabVIEW can use to keep the model in memory to prevent reloading it each time.
    """
    def __init__(self, modelpath=Path):
        self.modelpath = modelpath
        print(f"Loading model from : {modelpath}")
        self.model, self.modelpsfgen = experimental.preloadmodel(Path(modelpath))
