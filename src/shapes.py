import raster_geometry as rg
from skimage import measure
import matplotlib.pyplot as plt
from tifffile import imread, imsave

import logging
logger = logging.getLogger('')


def test(image_size=(64, 64, 64)):
    point = rg.sphere(shape=image_size, radius=1, position=.5)
    sphere1 = rg.sphere(shape=image_size, radius=3, position=[.25, .2, .3])
    sphere2 = rg.sphere(shape=image_size, radius=10, position=[.75, .8, .8])
    cube = rg.cuboid(shape=image_size, semisides=[1, 5, 5], position=[.4, .5, .8])
    cuboid = rg.cuboid(shape=image_size, semisides=[2, 4, 4], position=[.1, .2, .5])
    rhomboid = rg.rhomboid(shape=image_size, semidiagonals=[20, 1, 10], position=[.7, .8, .2])
    cylinder = rg.cylinder(shape=image_size, height=10, radius=2, position=[.9, .1, .1])
    return point + sphere1 + sphere2 + cube + cuboid + rhomboid + cylinder


def single_point(image_size=(64, 64, 64)):
    return rg.sphere(shape=image_size, radius=1, position=.5)


def two_points(image_size=(64, 64, 64)):
    p1 = rg.sphere(shape=image_size, radius=1, position=.2)
    p2 = rg.sphere(shape=image_size, radius=1, position=.8)
    return p1 + p2


def several_points(image_size=(64, 64, 64)):
    center = rg.sphere(shape=image_size, radius=1, position=.5)
    c1 = rg.sphere(shape=image_size, radius=1, position=.25)
    c2 = rg.sphere(shape=image_size, radius=1, position=[.25, .25, .75])
    c3 = rg.sphere(shape=image_size, radius=1, position=.75)
    c4 = rg.sphere(shape=image_size, radius=1, position=[.75, .75, .25])
    return center + c1 + c2 + c3 + c4


def sheet(image_size=(64, 64, 64)):
    return rg.rhomboid(shape=image_size, semidiagonals=[40, 1, 40], position=[.4, .5, .4])


def cube(image_size=(64, 64, 64)):
    return rg.cuboid(shape=image_size, semisides=[5, 5, 5], position=[.4, .5, .8])


def cylinder(image_size=(64, 64, 64)):
    return rg.cylinder(shape=image_size, height=60, radius=2, position=[.9, .1, .1])


def point_and_sheet(image_size=(64, 64, 64)):
    p = rg.sphere(shape=image_size, radius=1, position=.75)
    sheet = rg.cuboid(shape=image_size, semisides=[15, 5, 1], position=[.4, .5, .4])
    return p + sheet


def point_and_cylinder(image_size=(64, 64, 64)):
    p = rg.sphere(shape=image_size, radius=1, position=[.75, .75, .25])
    cylinder = rg.cylinder(shape=image_size, height=30, radius=2, position=[.2, .5, .6])
    return p + cylinder


def several_points_and_sheet(image_size=(64, 64, 64)):
    c1 = rg.sphere(shape=image_size, radius=1, position=[.3, .1, .7])
    c2 = rg.sphere(shape=image_size, radius=1, position=.75)
    c3 = rg.sphere(shape=image_size, radius=1, position=[.2, .6, .9])
    sheet = rg.rhomboid(shape=image_size, semidiagonals=[1, 10, 20], position=[.4, .5, .4])
    return c1 + c2 + c3 + sheet


def several_points_and_cylinder(image_size=(64, 64, 64)):
    c1 = rg.sphere(shape=image_size, radius=1, position=.2)
    c2 = rg.sphere(shape=image_size, radius=1, position=[.7, .8, .6])
    c3 = rg.sphere(shape=image_size, radius=1, position=[.6, .4, .2])
    cylinder = rg.cylinder(shape=image_size, height=10, radius=2, position=[.4, .5, .6])
    return c1 + c2 + c3 + cylinder


def simobjects(codename=None, image_size=(64, 64, 64), plot=False):

    hashtbl = {
        'test': test,
        'single_point': single_point,
        'two_points': two_points,
        'several_points': several_points,
        'cube': cube,
        'sheet': sheet,
        'cylinder': cylinder,
        'point_and_sheet': point_and_sheet,
        'point_and_cylinder': point_and_cylinder,
        'several_points_and_sheet': several_points_and_sheet,
        'several_points_and_cylinder': several_points_and_cylinder,
    }

    if codename is None:
        for s, func in hashtbl.items():
            obj = func(image_size=image_size)
            imsave(f"../data/shapes/{s}.tif", obj)
            logger.info(f"../data/shapes/{s}.tif")
    else:
        try:
            func = hashtbl[codename]
        except KeyError:
            raise ValueError('invalid codename')

        img = func(image_size=image_size)

        if plot:
            plot_3d_object(img)


def plot_3d_object(img):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

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
    plt.show()
