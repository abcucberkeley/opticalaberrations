
import sys
import logging

import matplotlib.pyplot as plt
import ujson
import numpy as np
from pathlib import Path
from tifffile import imsave
import raster_geometry as rg

from skimage.filters import scharr
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import skeletonize
from skimage.filters.rank import entropy


from wavefront import Wavefront
from synthetic import SyntheticPSF
from preprocessing import remove_background_noise, prep_sample
from utils import peak2valley, fftconvolution, resize_with_crop_or_pad
from vis import plot_wavefront
from embeddings import fft, ifft, compute_emb, fourier_embeddings, plot_embeddings

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def beads(
    gen: SyntheticPSF,
    object_size: float = 0,
    num_objs: int = 1,
    radius: float = .4,
):
    reference = np.zeros(gen.psf_shape)

    for i in range(num_objs):
        if object_size > 0:
            reference += rg.sphere(
                shape=gen.psf_shape,
                radius=object_size,
                position=np.random.uniform(low=.2, high=.8, size=3)
            ).astype(np.float) * np.random.random()
        else:
            if radius > 0:
                reference[
                    np.random.randint(int(gen.psf_shape[0] * (.5 - radius)), int(gen.psf_shape[0] * (.5 + radius))),
                    np.random.randint(int(gen.psf_shape[1] * (.5 - radius)), int(gen.psf_shape[1] * (.5 + radius))),
                    np.random.randint(int(gen.psf_shape[2] * (.5 - radius)), int(gen.psf_shape[2] * (.5 + radius)))
                ] += np.random.random()
            else:
                reference[gen.psf_shape[0] // 2, gen.psf_shape[1] // 2, gen.psf_shape[2] // 2] += np.random.random()

    reference /= np.max(reference)
    return reference


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
        nas=(.55, .65, .75, .85, .95, 1.),
        vcolorbar=True,
    )
    plt.savefig(f"{outdir}/{filename}_wavefront.png", bbox_inches='tight', pad_inches=.25)
    imsave(f"{outdir}/{filename}_wavefront.tif", wavefront.wave(size=128))

    # aberrated PSF without noise
    psf, amps, estsnr, maxcounts, lls_defocus_offset = gen.single_psf(
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
        maxcounts = np.max(noisy_img)
        psnr = np.sqrt(maxcounts)
    else:
        maxcounts = np.max(img)
        psnr = np.sqrt(maxcounts)
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
        embeddings = fourier_embeddings(
            inputs=noisy_img,
            iotf=gen.iotf,
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

    return noisy_img


if __name__ == "__main__":
    modes = 15
    x_voxel_size = .108
    y_voxel_size = .108
    z_voxel_size = .2
    lam_detection = .510
    psf_type = '../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
    input_shape = (64, 64, 64)
    na_detection = 1.0
    refractive_index = 1.33
    snr = 1000
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
    filename = f'mode_7'
    zernikes[7] = .2  # mu rms

    num_objs = 25
    reference = beads(gen=gen, object_size=0, num_objs=num_objs)

    wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)

    psf = create_synthetic_sample(
        wavefront=wavefront,
        filename=filename,
        outdir=outdir,
        gen=gen,
        noise=noise,
    )

    inputs = fftconvolution(sample=reference, kernel=psf)

    rand_noise = gen._random_noise(
        image=inputs,
        mean=gen.mean_background_noise,
        sigma=gen.sigma_background_noise
    )
    inputs *= snr ** 2
    inputs += rand_noise

    inputs = prep_sample(
        inputs,
        sample_voxel_size=gen.voxel_size,
        model_voxel_size=gen.voxel_size,
        remove_background=True,
        normalize=True,
        edge_filter=False,
    )

    fourier_embeddings(
        inputs=inputs,
        iotf=gen.iotf,
        plot=outdir / f"fourier_embeddings_num_objs_{num_objs}",
        embedding_option=gen.embedding_option,
        remove_interference=True,
    )

    masked_inputs = prep_sample(
        inputs,
        sample_voxel_size=gen.voxel_size,
        model_voxel_size=gen.voxel_size,
        remove_background=False,
        normalize=True,
        edge_filter=True,
    )

    fourier_embeddings(
        inputs=masked_inputs,
        iotf=gen.iotf,
        plot=outdir / f"masked_fourier_embeddings_num_objs_{num_objs}",
        embedding_option=gen.embedding_option,
        remove_interference=True,
    )
