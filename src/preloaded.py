from pathlib import Path
from backend import load, load_metadata
import numpy as np
from typing import Optional, Union
import tensorflow as tf


class Preloadedmodelclass:
    """ A class that LabVIEW can use to keep the model in memory to prevent reloading it each time.
    LabVIEW hates **kwargs.  So don't do that.
    """
    def __init__(
            self,
            modelpath: Path,
            ideal_empirical_psf: Optional[Union[Path, np.ndarray]] = None,
            ideal_empirical_psf_voxel_size: Optional[tuple] = None,
            n_modes: Optional[int] = None,
            psf_type: Optional[Union[Path, str]] = None
    ):
        
        if ideal_empirical_psf == "None" or ideal_empirical_psf is None:
            self.ideal_empirical_psf = None
            self.ideal_empirical_psf_voxel_size = None
        elif isinstance(ideal_empirical_psf, np.ndarray):          
            self.ideal_empirical_psf = ideal_empirical_psf
            self.ideal_empirical_psf_voxel_size = ideal_empirical_psf_voxel_size
        else:
            # needed to make sure this gets typed to Path, because comparison of str and Path will fail
            self.ideal_empirical_psf = Path(ideal_empirical_psf)
            self.ideal_empirical_psf_voxel_size = ideal_empirical_psf_voxel_size

        self.modelpath = Path(modelpath)

        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu_instance in physical_devices:
            tf.config.experimental.set_memory_growth(gpu_instance, True)

        self.modelpsfgen = load_metadata(self.modelpath, psf_type=psf_type, n_modes=n_modes)
        self.model = load(self.modelpath, mosaic=True)

        if self.ideal_empirical_psf is not None:
            self.modelpsfgen.update_ideal_psf_with_empirical(
                ideal_empirical_psf=self.ideal_empirical_psf,
                voxel_size=self.ideal_empirical_psf_voxel_size,
                remove_background=True,
                normalize=True,
            )
