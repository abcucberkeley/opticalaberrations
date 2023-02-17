
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
from embeddings import fourier_embeddings, plot_embeddings

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
    x_voxel_size = .104
    y_voxel_size = .104
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
    filename = f'ideal'
    zernikes[7] = 0  # mu rms

    wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)

    psf = create_synthetic_sample(
        wavefront=wavefront,
        filename=filename,
        outdir=outdir,
        gen=gen,
        noise=noise,
    )

    embeddings_psf = fourier_embeddings(
        inputs=psf,
        iotf=gen.iotf,
        plot=outdir / f"psf",
        remove_interference=False,
    )

    # num_objs = 50
    # reference = beads(gen=gen, object_size=0, num_objs=num_objs)
    #
    # zernikes = np.zeros(modes)
    # zernikes[5] = .05  # mu rms
    # zernikes[11] = .05  # mu rms
    # wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    # f1 = gen.single_psf(wavefront, normed=True, noise=False)
    #
    # if num_objs > 1:
    #     f1 = fftconvolution(sample=reference, kernel=f1)
    #
    # embeddings_f1 = fourier_embeddings(
    #     f1,
    #     iotf=gen.iotf,
    #     plot=outdir / f"f1_num_objs_{num_objs}",
    #     remove_interference=False,
    # )
    #
    # zernikes = np.zeros(modes)
    # zernikes[5] = -.05  # mu rms
    # zernikes[11] = .05  # mu rms
    # wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    # f2 = gen.single_psf(wavefront, normed=True, noise=False)
    #
    # if num_objs > 1:
    #     f2 = fftconvolution(sample=reference, kernel=f2)
    #
    # embeddings_f2 = fourier_embeddings(
    #     f2,
    #     iotf=gen.iotf,
    #     plot=outdir / f"f2_num_objs_{num_objs}",
    #     remove_interference=False,
    # )
    #
    # zernikes = np.zeros(modes)
    # zernikes[11] = .05  # mu rms
    # wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    # psf = gen.single_psf(wavefront, normed=True, noise=False)
    # embeddings_psf = fourier_embeddings(
    #     psf,
    #     iotf=gen.iotf,
    #     plot=outdir / f"psf_num_objs_{num_objs}",
    #     remove_interference=False,
    # )
    #
    # if num_objs > 1:
    #     psf = fftconvolution(sample=reference, kernel=psf)
    #
    # ratio = gen.fft(f1) / gen.fft(f2)
    # pseudo_psf = np.abs(gen.ifft(ratio))
    # pseudo_psf /= np.nanmax(pseudo_psf)
    # pseudo_psf = resize_with_crop_or_pad(pseudo_psf, crop_shape=(32, 32, 32))
    #
    # alpha = gen.compute_emb(
    #     ratio,
    #     val='abs',
    #     ratio=True,
    #     norm=True,
    #     embedding_option='spatial_planes',
    # )
    #
    # phi = gen.compute_emb(
    #     ratio,
    #     val='angle',
    #     ratio=False,
    #     na_mask=True,
    #     norm=False,
    #     embedding_option='spatial_planes',
    # )
    # ratio_emb = np.concatenate([alpha, phi], axis=0)
    #
    # plot_embeddings(
    #     inputs=pseudo_psf,
    #     emb=ratio_emb,
    #     save_path=outdir / f"pseudo_psf_num_objs_{num_objs}",
    # )
    #
    # zernikes = np.zeros(modes)
    # zernikes[11] = .05  # mu rms
    # wavefront = Wavefront(zernikes, lam_detection=gen.lam_detection)
    # img = gen.single_psf(wavefront, normed=True, noise=False)
    #
    # if num_objs > 1:
    #     img = fftconvolution(sample=reference, kernel=img)
    #
    # embeddings_img = fourier_embeddings(
    #     img,
    #     iotf=gen.iotf,
    #     plot=outdir / f"img_num_objs_{num_objs}",
    #     remove_interference=False,
    # )
    #
    # from skimage.feature import hog
    #
    # structure = np.zeros_like(img)
    # for plane in range(img.shape[0]):
    #     fd, hog_image = hog(
    #         img[plane],
    #         orientations=9,
    #         pixels_per_cell=(3, 3),
    #         cells_per_block=(3, 3),
    #         visualize=True,
    #         block_norm='L2-Hys'
    #     )
    #     structure[plane] = hog_image
    #
    #
    # # t_loc_otsu = rank.otsu(img, ball(15))
    # # t_glob_otsu = threshold_otsu(img)
    # # structure = np.zeros_like(img)
    # # structure[img >= t_glob_otsu] = img[img >= t_glob_otsu]**3
    # structure /= np.nanmax(structure)
    #
    # # if num_objs > 1:
    # #     structure = fftconvolution(sample=structure, kernel=gen.ipsf)
    #
    # embeddings_structure = fourier_embeddings(
    #     structure,
    #     iotf=gen.iotf,
    #     plot=outdir / f"structure_num_objs_{num_objs}",
    #     remove_interference=False,
    # )
    #
    # ratio = gen.fft(img) / gen.fft(structure)
    # reconstructed_psf = np.abs(gen.ifft(ratio))
    # reconstructed_psf /= np.nanmax(reconstructed_psf)
    #
    # alpha = gen.compute_emb(
    #     ratio,
    #     val='abs',
    #     ratio=True,
    #     norm=True,
    #     embedding_option='spatial_planes',
    # )
    #
    # phi = gen.compute_emb(
    #     ratio,
    #     val='angle',
    #     ratio=False,
    #     na_mask=True,
    #     norm=False,
    #     embedding_option='spatial_planes',
    # )
    # ratio_emb = np.concatenate([alpha, phi], axis=0)
    #
    # plot_embeddings(
    #     inputs=reconstructed_psf,
    #     emb=ratio_emb,
    #     save_path=outdir / f"fourier_embeddings_num_objs_{num_objs}",
    # )
