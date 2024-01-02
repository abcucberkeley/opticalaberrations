# phase retrieve a pupil
from pathlib import Path
import time
import warnings
import tifffile as tif
import pyotf.phaseretrieval as pr
from pyotf.utils import prep_data_for_PR
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# read in data from fixtures
if __name__ == '__main__':
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.int_(tif.imread(Path(r"C:\SPIM\SPIM\SPIM Support files\Matlab Phase Retrieval\PSF measurement.tif")))
        
        # prep data
    data_prepped = prep_data_for_PR(data,  multiplier=1.1)

    # set up model params
    params = dict(
        wl=515, 
        na=1.00,
        ni=1.33,
        res=97,
        zres=100,
        )

    # retrieve the phase
    pr_start = time.time()
    print("Starting phase retrieval ... ", end="", flush=True)
    pr_result = pr.retrieve_phase(data_prepped, params, max_iters=3000, pupil_tol=1e-5, mse_tol=1e-5, phase_only=True)
    pr_time = time.time() - pr_start
    print(f"{pr_time:.1f} seconds were required to retrieve the pupil function")

    # plot
    pr_result.plot()
    pr_result.plot_convergence()

    # fit to zernikes
    zd_start = time.time()
    print("Starting zernike decomposition ... ", end="", flush=True)
    pr_result.fit_to_zernikes(15, mapping=pr.osa2degrees)
    #pr_result.zd_result.plot()
    zd_time = time.time() - zd_start
    print(f"{zd_time:.1f} seconds were required to fit 120 Zernikes")

    # plot
    #pr_result.zd_result.plot_named_coefs()
    pr_result.zd_result.plot_coefs()

    # show
    plt.show()
    print("Done.")