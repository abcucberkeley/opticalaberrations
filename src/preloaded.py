from pathlib import Path
from tensorflow import config as tfc
from backend import load, load_metadata


class Preloadedmodelclass:
    """ A class that LabVIEW can use to keep the model in memory to prevent reloading it each time.
    LabVIEW hates **kwargs.  So don't do that.
    """
    def __init__(self, modelpath: Path, ideal_empirical_psf=None, ideal_empirical_psf_voxel_size=None):
        
        if ideal_empirical_psf == "None" or ideal_empirical_psf is None:
            self.ideal_empirical_psf = None
            self.ideal_empirical_psf_voxel_size = None
        else:
            self.ideal_empirical_psf = ideal_empirical_psf
            self.ideal_empirical_psf_voxel_size = ideal_empirical_psf_voxel_size

        self.modelpath = Path(modelpath)
                
        physical_devices = tfc.list_physical_devices('GPU')
        for gpu_instance in physical_devices:
            tfc.experimental.set_memory_growth(gpu_instance, True)

        self.modelpsfgen = load_metadata(self.modelpath)
        self.model = load(self.modelpath, mosaic=True)

        if self.ideal_empirical_psf is not None:
            self.modelpsfgen.update_ideal_psf_with_empirical(
                ideal_empirical_psf=self.ideal_empirical_psf,
                voxel_size=self.ideal_empirical_psf_voxel_size,
                remove_background=True,
                normalize=True,
            )
