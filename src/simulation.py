
import sys
import logging

import matplotlib.pyplot as plt
import ujson
import numpy as np
from pathlib import Path
from tifffile import imsave

from wavefront import Wavefront
from synthetic import SyntheticPSF
from preprocessing import remove_background_noise
from utils import peak2valley
from vis import plot_wavefront

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_synthetic_sample(savepath, inputs, amps, snr, maxcounts, p2v, npoints=1):

    logger.info(f"Saved: {savepath}")
    imsave(f"{savepath}.tif", inputs)

    with Path(f"{savepath}.json").open('w') as f:
        json = dict(
            path=f"{savepath}.tif",
            zernikes=amps.tolist(),
            snr=int(snr),
            shape=inputs.shape,
            maxcounts=int(maxcounts),
            npoints=int(npoints),
            peak2peak=float(p2v)
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def create_synthetic_sample(
    wavefront: Wavefront,
    filename: str,
    outdir: Path,
    gen: SyntheticPSF,
    noise: bool = True,
    embedding_option: list = ('principle_planes', 'spatial_planes'),
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    remove_interference: bool = False
):
    outdir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots()
    plot_wavefront(
        ax,
        wavefront.wave(100),
        label=filename,
        nas=(.55, .65, .75, .85, .95),
        vcolorbar=True,
    )
    plt.savefig(f"{outdir}/{filename}_wavefront.png", bbox_inches='tight', pad_inches=.25)

    # aberrated PSF without noise
    psf, amps, phi, maxcounts = gen.single_psf(
        wavefront,
        normed=True,
        noise=False,
        meta=True,
    )
    snr = gen._randuniform(gen.snr)
    img = psf * snr ** 2

    if noise:
        rand_noise = gen._random_noise(
            image=img,
            mean=gen.mean_background_noise,
            sigma=gen.sigma_background_noise
        )
        noisy_img = rand_noise + img
        psnr = np.sqrt(np.max(noisy_img))
        maxcounts = np.max(noisy_img)
    else:
        psnr = np.mean(np.array(snr))
        maxcounts = np.max(img)
        noisy_img = img

    save_synthetic_sample(
        outdir / filename,
        noisy_img,
        amps=amps,
        snr=psnr,
        maxcounts=maxcounts,
        p2v=peak2valley(amps, wavelength=gen.lam_detection)
    )

    if noise:
        noisy_img = remove_background_noise(noisy_img)

    noisy_img /= np.max(noisy_img)

    for e in set(embedding_option):
        embeddings = gen.embedding(
            psf=noisy_img,
            remove_interference=remove_interference,
            embedding_option=e,
            alpha_val=alpha_val,
            phi_val=phi_val,
            plot=outdir/f"{filename}_{e}"
        )

        save_synthetic_sample(
            outdir/f"{filename}_{e}",
            embeddings,
            amps=amps,
            snr=psnr,
            maxcounts=maxcounts,
            p2v=peak2valley(amps, wavelength=gen.lam_detection)
        )


if __name__ == "__main__":
    modes = 15
    x_voxel_size = .108
    y_voxel_size = .108
    z_voxel_size = .108
    lam_detection = .510
    psf_type = 'widefield'
    input_shape = (64, 64, 64)
    na_detection = 1.0
    refractive_index = 1.33
    snr = 100
    noise = False
    outdir = Path('../test_samples/')
    zernikes = np.zeros(modes)

    gen = SyntheticPSF(
        order='ansi',
        n_modes=modes,
        snr=snr,
        psf_type=psf_type,
        lam_detection=lam_detection,
        psf_shape=input_shape,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        refractive_index=refractive_index,
        na_detection=na_detection,
    )

    # ANSI index
    filename = f'mode_12'
    zernikes[3] = .05  # mu rms
    zernikes[12] = .05  # mu rms

    wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)

    create_synthetic_sample(
        wavefront=wavefront,
        filename=filename,
        outdir=outdir,
        gen=gen,
        noise=noise,
    )
