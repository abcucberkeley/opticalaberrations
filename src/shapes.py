
import numpy as np
import raster_geometry as rg
from skimage import measure
import matplotlib.pyplot as plt
from tifffile import imread, imsave

import logging
logger = logging.getLogger('')


def single_point(image_size=(64, 64, 64)):
    img = np.zeros(image_size)
    img[64, 64, 64] = 1.
    return img.astype(np.float)


def two_points(image_size=(64, 64, 64)):
    img = np.zeros(image_size)
    img[12, 12, 12] = 1.
    img[52, 52, 52] = 1.
    return img.astype(np.float)


def several_points(image_size=(64, 64, 64)):
    img = np.zeros(image_size)
    img[16, 16, 16] = 1.
    img[32, 32, 32] = 1.
    img[48, 48, 48] = 1.
    img[16, 16, 48] = 1.
    img[48, 48, 16] = 1.
    return img.astype(np.float)


def line(image_size=(64, 64, 64)):
    img = np.zeros(image_size)
    img[20:50, 32, 32] = 1.
    return img.astype(np.float)


def sphere(image_size=(64, 64, 64)):
    img = rg.sphere(shape=image_size, radius=.5, position=.5)
    return img.astype(np.float)


def sheet(image_size=(64, 64, 64)):
    img = rg.rhomboid(shape=image_size, semidiagonals=[40, 1, 40], position=[.4, .5, .4])
    return img.astype(np.float)


def cylinder(image_size=(64, 64, 64)):
    img = rg.cylinder(shape=image_size, height=60, radius=2, position=[.9, .1, .1])
    return img.astype(np.float)


def point_and_line(image_size=(64, 64, 64)):
    img = np.zeros(image_size)
    img[48, 48, 48] = 1.
    img[24, 20:50, 24] = 1.
    return img.astype(np.float)


def point_and_sheet(image_size=(64, 64, 64)):
    p = np.zeros(image_size)
    p[48, 48, 48] = 1.
    sh = rg.cuboid(shape=image_size, semisides=[15, 5, 1], position=[.4, .5, .4]).astype(np.float)
    return p + sh


def point_and_cylinder(image_size=(64, 64, 64)):
    p = np.zeros(image_size)
    p[48, 48, 16] = 1.
    cc = rg.cylinder(shape=image_size, height=30, radius=2, position=[.2, .5, .6]).astype(np.float)
    return p + cc


def several_points_and_line(image_size=(64, 64, 64)):
    img = np.zeros(image_size)
    img[12, 12, 12] = 1.
    img[48, 48, 12] = 1.
    img[48, 48, 48] = 1.
    img[32, 32, 20:50] = 1.
    return img.astype(np.float)


def several_points_and_sheet(image_size=(64, 64, 64)):
    ps = np.zeros(image_size)
    ps[20, 5, 40] = 1.
    ps[48, 48, 48] = 1.
    ps[12, 40, 60] = 1.
    sh = rg.rhomboid(shape=image_size, semidiagonals=[1, 10, 20], position=[.4, .5, .4]).astype(np.float)
    return ps + sh


def several_points_and_cylinder(image_size=(64, 64, 64)):
    ps = np.zeros(image_size)
    ps[12, 12, 12] = 1.
    ps[45, 50, 40] = 1.
    ps[40, 25, 12] = 1.
    cc = rg.cylinder(shape=image_size, height=10, radius=2, position=[.4, .5, .6]).astype(np.float)
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

    hashtbl = {
        'single_point': single_point,
        'two_points': two_points,
        'several_points': several_points,
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
