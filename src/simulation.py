
import sys
import logging

import matplotlib.pyplot as plt
import ujson
import numpy as np
from pathlib import Path
from tifffile import imsave
import raster_geometry as rg

from wavefront import Wavefront
from synthetic import SyntheticPSF
from preprocessing import remove_background_noise
from utils import peak2valley, fftconvolution, resize_with_crop_or_pad
from vis import plot_wavefront

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
        nas=(.55, .65, .75, .85, .95),
        vcolorbar=True,
    )
    plt.savefig(f"{outdir}/{filename}_wavefront.png", bbox_inches='tight', pad_inches=.25)
    imsave(f"{outdir}/{filename}_wavefront.tif", wavefront.wave(size=128))

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

    return noisy_img


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
    # filename = f'mode_7'
    # zernikes[7] = .1  # mu rms
    #
    # wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    #
    # psf = create_synthetic_sample(
    #     wavefront=wavefront,
    #     filename=filename,
    #     outdir=outdir,
    #     gen=gen,
    #     noise=noise,
    # )
    #
    # embeddings_psf = gen.embedding(
    #     psf=psf,
    #     plot=outdir / f"psf",
    #     remove_interference=False,
    # )

    num_objs = 50
    reference = beads(gen=gen, object_size=0, num_objs=num_objs)

    zernikes = np.zeros(modes)
    zernikes[5] = .05  # mu rms
    zernikes[11] = .05  # mu rms
    wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    f1 = gen.single_psf(wavefront, normed=True, noise=False)

    if num_objs > 1:
        f1 = fftconvolution(sample=reference, kernel=f1)

    embeddings_f1 = gen.embedding(
        psf=f1,
        plot=outdir / f"f1_num_objs_{num_objs}",
        remove_interference=False,
    )

    zernikes = np.zeros(modes)
    zernikes[5] = -.05  # mu rms
    zernikes[11] = .05  # mu rms
    wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    f2 = gen.single_psf(wavefront, normed=True, noise=False)

    if num_objs > 1:
        f2 = fftconvolution(sample=reference, kernel=f2)

    embeddings_f2 = gen.embedding(
        psf=f2,
        plot=outdir / f"f2_num_objs_{num_objs}",
        remove_interference=False,
    )

    zernikes = np.zeros(modes)
    zernikes[11] = .05  # mu rms
    wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    psf = gen.single_psf(wavefront, normed=True, noise=False)
    embeddings_psf = gen.embedding(
        psf=psf,
        plot=outdir / f"psf_num_objs_{num_objs}",
        remove_interference=False,
    )

    if num_objs > 1:
        psf = fftconvolution(sample=reference, kernel=psf)

    ratio = gen.fft(f1) / gen.fft(f2)
    pseudo_psf = np.abs(gen.ifft(ratio))
    pseudo_psf /= np.nanmax(pseudo_psf)
    pseudo_psf = resize_with_crop_or_pad(pseudo_psf, crop_shape=(32, 32, 32))

    alpha = gen.compute_emb(
        ratio,
        val='abs',
        ratio=True,
        norm=True,
        embedding_option='spatial_planes',
    )

    phi = gen.compute_emb(
        ratio,
        val='angle',
        ratio=False,
        na_mask=True,
        norm=False,
        embedding_option='spatial_planes',
    )
    ratio_emb = np.concatenate([alpha, phi], axis=0)

    gen.plot_embeddings(
        inputs=pseudo_psf,
        emb=ratio_emb,
        save_path=outdir / f"fourier_embeddings_num_objs_{num_objs}",
    )

