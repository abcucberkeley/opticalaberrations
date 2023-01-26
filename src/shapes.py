import matplotlib
matplotlib.use('Agg')

from functools import partial

import numpy as np
import raster_geometry as rg
from skimage import measure, transform
import matplotlib.pyplot as plt
from tifffile import imread, imsave
from scipy import signal
from pathlib import Path
from tqdm import trange

import backend
from synthetic import SyntheticPSF
from wavefront import Wavefront

import logging
logger = logging.getLogger('')


def single_point(image_size):
    img = np.zeros(image_size)
    img[image_size[0]//2, image_size[1]//2, image_size[2]//2] = 1.
    return img.astype(np.float)


def several_points(image_size, npoints=100, radius=.2):
    img = np.zeros(image_size)
    for i in range(npoints):
        img[
            np.random.randint(int(image_size[0]*(.5 - radius)), int(image_size[0]*(.5 + radius))),
            np.random.randint(int(image_size[1]*(.5 - radius)), int(image_size[1]*(.5 + radius))),
            np.random.randint(int(image_size[2]*(.5 - radius)), int(image_size[2]*(.5 + radius)))
        ] = 1.

    return img.astype(np.float)


def line(image_size):
    img = np.zeros(image_size)
    img[int(image_size[0]*.3):int(image_size[0]*.7), image_size[1]//2, image_size[2]//2] = 1.
    return img.astype(np.float)


def several_lines(image_size, nlines=10, radius=.2):
    img = np.zeros(image_size)
    for i in range(nlines):
        ax = np.random.choice(['x', 'y', 'z'])
        if ax == 'x':
            img[
                np.random.randint(int(image_size[0] * (.5-radius)), int(image_size[0] * (.5+radius))),
                np.random.randint(int(image_size[1]*(.5-radius)), int(image_size[1]*(.5+radius))),
                np.random.randint(25, image_size[2]//2):np.random.randint(image_size[2]//2, image_size[0]-25),
            ] = 1.
        elif ax == 'y':
            img[
                np.random.randint(int(image_size[0] * (.5-radius)), int(image_size[0] * (.5+radius))),
                np.random.randint(25, image_size[1]//2):np.random.randint(image_size[1]//2, image_size[0]-25),
                np.random.randint(int(image_size[2] * (.5-radius)), int(image_size[2] * (.5+radius)))
            ] = 1.
        else:
            img[
                np.random.randint(25, image_size[0]//2):np.random.randint(image_size[0]//2, image_size[0]-25),
                np.random.randint(int(image_size[1] * (.5-radius)), int(image_size[1] * (.5+radius))),
                np.random.randint(int(image_size[2] * (.5-radius)), int(image_size[2] * (.5+radius)))
            ] = 1.

    return img.astype(np.float)


def sphere(image_size):
    img = rg.sphere(shape=image_size, radius=.5, position=.5)
    return img.astype(np.float)


def sheet(image_size):
    img = rg.rhomboid(
        shape=image_size,
        semidiagonals=[int(.5*image_size[0]), 1, int(.5*image_size[2])],
        position=[.4, .5, .4])
    return img.astype(np.float)


def cylinder(image_size):
    img = rg.cylinder(shape=image_size, height=int(.5*image_size[0]), radius=2, position=[.5, .5, .5])
    return img.astype(np.float)


def point_and_line(image_size):
    img = np.zeros(image_size)
    img[image_size[0]//2, image_size[1]//2, image_size[2]//2] = 1.
    img[image_size[0]//4, image_size[1]//6:image_size[1]//3, image_size[2]//4] = 1.
    return img.astype(np.float)


def point_and_sheet(image_size):
    p = np.zeros(image_size)
    p[int(image_size[0]//1.2), int(image_size[1]//1.2), int(image_size[2]//1.2)] = 1.
    sh = rg.cuboid(
        shape=image_size,
        semisides=[int(.25*image_size[0]), int(.05*image_size[1]), 1],
        position=[.4, .5, .4]
    ).astype(np.float)
    return p + sh


def point_and_cylinder(image_size):
    p = np.zeros(image_size)
    p[int(image_size[0]//1.2), int(image_size[1]//1.2), image_size[2]//4] = 1.
    cc = rg.cylinder(
        shape=image_size,
        height=int(.3*image_size[0]),
        radius=int(.1*image_size[0]),
        position=[.2, .5, .6]
    ).astype(np.float)
    return p + cc


def several_points_and_line(image_size):
    img = np.zeros(image_size)
    for i in range(25):
        img[
            np.random.randint(int(image_size[0]*.3), int(image_size[0]*.7)),
            np.random.randint(int(image_size[1]*.3), int(image_size[1]*.7)),
            np.random.randint(int(image_size[2]*.3), int(image_size[2]*.7))
        ] = 1.

    img[image_size[0]//2, image_size[1]//2, image_size[2]//6:image_size[2]//3] = 1.
    return img.astype(np.float)


def several_points_and_sheet(image_size):
    ps = np.zeros(image_size)
    for i in range(25):
        ps[
            np.random.randint(int(image_size[0]*.3), int(image_size[0]*.7)),
            np.random.randint(int(image_size[1]*.3), int(image_size[1]*.7)),
            np.random.randint(int(image_size[2]*.3), int(image_size[2]*.7))
        ] = 1.

    sh = rg.rhomboid(
        shape=image_size,
        semidiagonals=[1, int(.1*image_size[1]), int(.2*image_size[2])],
        position=[.4, .5, .4]
    ).astype(np.float)
    return ps + sh


def several_points_and_cylinder(image_size):
    ps = np.zeros(image_size)
    for i in range(25):
        ps[
            np.random.randint(int(image_size[0]*.3), int(image_size[0]*.7)),
            np.random.randint(int(image_size[1]*.3), int(image_size[1]*.7)),
            np.random.randint(int(image_size[2]*.3), int(image_size[2]*.7))
        ] = 1.

    cc = rg.cylinder(
        shape=image_size,
        height=int(.1*image_size[0]),
        radius=int(.05*image_size[0]),
        position=[.4, .5, .6]
    ).astype(np.float)
    return ps + cc


def plot_3d_object(img, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    try:
        verts, faces, normals, values = measure.marching_cubes(img)
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            cmap='nipy_spectral',
            antialiased=False,
            linewidth=0.0
        )
        ax.set_xlim(0, img.shape[0])
        ax.set_xlabel('X')
        ax.set_ylim(0, img.shape[1])
        ax.set_ylabel('Y')
        ax.set_zlim(0, img.shape[2])
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    except RuntimeError:
        pass


def simobjects(codename=None, image_size=(256, 256, 256), plot=False):
    np.random.seed(202207)

    hashtbl = {
        'single_point': single_point,
        '2_points_10p_radius': partial(several_points, npoints=2, radius=.1),
        '5_points_10p_radius': partial(several_points, npoints=5, radius=.1),
        '10_points_10p_radius': partial(several_points, npoints=10, radius=.1),
        '25_points_10p_radius': partial(several_points, npoints=25, radius=.1),
        '50_points_10p_radius': partial(several_points, npoints=50, radius=.1),
        '2_points_20p_radius': partial(several_points, npoints=2, radius=.2),
        '5_points_20p_radius': partial(several_points, npoints=5, radius=.2),
        '10_points_20p_radius': partial(several_points, npoints=10, radius=.2),
        '25_points_20p_radius': partial(several_points, npoints=25, radius=.2),
        '50_points_20p_radius': partial(several_points, npoints=50, radius=.2),
        '2_points_30p_radius': partial(several_points, npoints=2, radius=.3),
        '5_points_30p_radius': partial(several_points, npoints=5, radius=.3),
        '10_points_30p_radius': partial(several_points, npoints=10, radius=.3),
        '25_points_30p_radius': partial(several_points, npoints=25, radius=.3),
        '50_points_30p_radius': partial(several_points, npoints=50, radius=.3),
        '2_points_50p_radius': partial(several_points, npoints=2, radius=.5),
        '5_points_50p_radius': partial(several_points, npoints=5, radius=.5),
        '10_points_50p_radius': partial(several_points, npoints=10, radius=.5),
        '25_points_50p_radius': partial(several_points, npoints=25, radius=.5),
        '50_points_50p_radius': partial(several_points, npoints=50, radius=.5),
        'line': line,
        'sheet': sheet,
        'sphere': sphere,
        'cylinder': cylinder,
        'point_and_line': point_and_line,
        'point_and_sheet': point_and_sheet,
        'point_and_cylinder': point_and_cylinder,
        'several_points_and_line': several_points_and_line,
        'several_points_and_sheet': several_points_and_sheet,
        'several_points_and_cylinder': several_points_and_cylinder,
    }

    if codename is None:
        for s, func in hashtbl.items():
            obj = func(image_size=image_size)
            imsave(f"../data/shapes/{s}.tif", obj)
            logger.info(f"../data/shapes/{s}.tif")

            if plot:
                plot_3d_object(obj, title=s)
    else:
        try:
            func = hashtbl[codename]
        except KeyError:
            raise ValueError('invalid codename')

        img = func(image_size=image_size)

        if plot:
            plot_3d_object(img, title=codename)


def similarity(
    image_size=(256, 256, 256),
    reference_voxel_size=(.15, .0375, .0375),
    radius=.1,
    nobjects=1,
    obj='lines',
):
    model = backend.load(Path(f'../models/new/embeddings/transformers/p32-p16-p8x2'))
    savepath = Path(f'../data/similarity/{image_size[0]}/radius_{radius}_n{obj}_{nobjects}')
    savepath.mkdir(exist_ok=True, parents=True)

    modelgen = SyntheticPSF(
        n_modes=55,
        lam_detection=.605,
        psf_shape=(64, 64, 64),
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        snr=100,
    )

    gen = SyntheticPSF(
        n_modes=55,
        lam_detection=.605,
        psf_shape=image_size,
        z_voxel_size=reference_voxel_size[0],
        y_voxel_size=reference_voxel_size[1],
        x_voxel_size=reference_voxel_size[2],
        snr=100,
    )

    ys = np.zeros(55)
    ys[10] = .1
    kernel = gen.single_psf(
        phi=Wavefront(ys, lam_detection=.605),
        normed=True,
        noise=False,
        meta=False
    )
    imsave(savepath / f'kernel.tif', kernel)

    embeddings = []
    for k in trange(25):
        if obj == 'points':
            reference = several_points(image_size, npoints=nobjects, radius=radius)
        else:
            reference = several_lines(image_size, nlines=nobjects, radius=radius)

        imsave(savepath / f'{k}_reference.tif', reference)

        conv = signal.convolve(reference, kernel, mode='same')
        imsave(savepath / f'{k}_convolved.tif', conv)

        preds, stds, embs = backend.bootstrap_predict(
            model,
            np.expand_dims(conv[np.newaxis, :], axis=-1),
            psfgen=modelgen,
            resize=reference_voxel_size,
            batch_size=1,
            n_samples=1,
            desc=f'Iter[{k}]',
            plot=savepath/f'{k}_embeddings',
            return_embeddings=True
        )
        embeddings.append(embs)

    embeddings = np.stack(embeddings, axis=0)
    embeddings = np.mean(embeddings, axis=0)
    imsave(savepath / f'embeddings_average.tif', embeddings)
