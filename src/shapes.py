from functools import partial

import numpy as np
import raster_geometry as rg
from skimage import measure
import matplotlib.pyplot as plt
from tifffile import imread, imsave

import logging
logger = logging.getLogger('')
np.random.seed(2022)


def single_point(image_size):
    img = np.zeros(image_size)
    img[image_size[0]//2, image_size[1]//2, image_size[2]//2] = 1.
    return img.astype(np.float)


def two_points(image_size):
    img = np.zeros(image_size)
    img[image_size[0]//2, image_size[1]//2, image_size[2]//2] = 1.
    img[image_size[0]//4, image_size[1]//4, image_size[2]//4] = 1.
    return img.astype(np.float)


def five_points(image_size):
    img = np.zeros(image_size)
    img[image_size[0]//2, image_size[1]//2, image_size[2]//2] = 1.
    img[image_size[0]//4, image_size[1]//4, image_size[2]//4] = 1.
    img[image_size[0]//3, int(image_size[1]//1.2), int(image_size[2]//1.2)] = 1.
    img[image_size[0]//4, image_size[1]//4, int(image_size[2]//1.2)] = 1.
    img[int(image_size[0]//1.2), int(image_size[1]//1.2), image_size[2]//4] = 1.
    return img.astype(np.float)


def several_points(image_size, npoints=100):
    img = np.zeros(image_size)
    for i in range(npoints):
        img[
            np.random.randint(int(image_size[0]*.4), int(image_size[0]*.6)),
            np.random.randint(int(image_size[1]*.4), int(image_size[1]*.6)),
            np.random.randint(int(image_size[2]*.4), int(image_size[2]*.6))
        ] = 1.

    return img.astype(np.float)


def line(image_size):
    img = np.zeros(image_size)
    img[int(image_size[0]*.4):int(image_size[0]*.6), image_size[1]//2, image_size[2]//2] = 1.
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
            np.random.randint(int(image_size[0]*.4), int(image_size[0]*.6)),
            np.random.randint(int(image_size[1]*.4), int(image_size[1]*.6)),
            np.random.randint(int(image_size[2]*.4), int(image_size[2]*.6))
        ] = 1.

    img[image_size[0]//2, image_size[1]//2, image_size[2]//6:image_size[2]//3] = 1.
    return img.astype(np.float)


def several_points_and_sheet(image_size):
    ps = np.zeros(image_size)
    for i in range(25):
        ps[
            np.random.randint(int(image_size[0]*.4), int(image_size[0]*.6)),
            np.random.randint(int(image_size[1]*.4), int(image_size[1]*.6)),
            np.random.randint(int(image_size[2]*.4), int(image_size[2]*.6))
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
            np.random.randint(int(image_size[0]*.4), int(image_size[0]*.6)),
            np.random.randint(int(image_size[1]*.4), int(image_size[1]*.6)),
            np.random.randint(int(image_size[2]*.4), int(image_size[2]*.6))
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


def simobjects(codename=None, image_size=(512, 512, 512), plot=True):

    hashtbl = {
        'single_point': single_point,
        'two_points': two_points,
        'five_points': five_points,
        '2_points': partial(several_points, npoints=2),
        '5_points': partial(several_points, npoints=5),
        '10_points': partial(several_points, npoints=10),
        '25_points': partial(several_points, npoints=25),
        '50_points': partial(several_points, npoints=50),
        '75_points': partial(several_points, npoints=75),
        '100_points': partial(several_points, npoints=100),
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
