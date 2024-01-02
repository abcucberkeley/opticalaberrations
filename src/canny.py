"""
# canny_edge_detector_3D
Perform Canny edge detection in 3D in python.

Based on the [step-by-step implementation of the Canny edge detection]
(https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123)
from @FienSoP, I derived a 3D version for application on 3D images.
This is comparable with [edge3](https://nl.mathworks.com/help/images/ref/edge3.html)
from the Image Processing Toolbox of Matlab.

canny_edge_detector_3D.py includes the actual edge detection.
slicer_3D.py is a useful tool to scroll through 3D images for visualizing your results.
Keep in mind that the edges go in three directions and only 2D images are displayed using the slicer.

The parameters like sigma, lowthresholdratio and highthresholdratio
can be set according to preferences of your image and application.

Canny_Edge_Detection_3D_Example_Notebook.ipynb is an example notebook you can use and adjust for your application.
For this specific example I made use of [this](https://figshare.com/s/2904b1ee61c3240f9291) openly available MRI dataset.
I used 'MRI_4.nii' for this case.

The results from the Notebook look like this:

![image](https://user-images.githubusercontent.com/93598891/142627673-6425a0ac-304d-4c6a-aff5-d55b0c86d69f.png)

Hopefully this tool is useful for others as well and it will save time writing it!

https://github.com/IlvaHou55/canny_edge_detector_3D
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve


class CannyEdgeDetector3D:
    def __init__(self, img, sigma=0.6, lowthresholdratio=0.3, highthresholdratio=0.2, weak_voxel=75, strong_voxel=255):
        self.img = img
        self.img_edges = []
        self.img_smoothed = None
        self.gradientMat = None
        self.phiMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.sigma = sigma
        self.lowThreshold = lowthresholdratio
        self.highThreshold = highthresholdratio
        self.weak_voxel = weak_voxel
        self.strong_voxel = strong_voxel
        return

    def sobel_filters(self, img):
        Kx = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], np.float32)
        Ky = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], np.float32)
        Kz = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                       [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]], np.float32)

        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)
        Iz = convolve(img, Kz)

        G = np.sqrt(Ix ** 2 + Iy ** 2 + Iz ** 2)
        G = G / G.max() * 255

        phi = np.arctan2(Iy, Ix)
        theta = np.arctan2(Iy, Iz)

        return (G, phi, theta)

    def non_max_suppression(self, img, phi, theta):
        M, N, O = img.shape
        Z = np.zeros((M, N, O), dtype=np.int32)
        phi[phi < 0] += np.pi
        theta[theta < 0] += np.pi

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                for k in range(1, O - 1):
                    try:
                        q = 255
                        r = 255

                        # theta 0
                        if (0 <= theta[i, j, k] < (0.125 * np.pi)) or ((0.875 * np.pi) <= theta[i, j, k] <= np.pi):
                            q = img[i, j, k + 1]
                            r = img[i, j, k - 1]

                        # theta 1/4 pi
                        elif ((0.125 * np.pi) <= theta[i, j, k] < (0.375 * np.pi)):
                            # phi 0
                            if (0 <= phi[i, j, k] < (0.125 * np.pi)) or ((0.875 * np.pi) <= phi[i, j, k] <= np.pi):
                                q = img[i + 1, j, k + 1]
                                r = img[i - 1, j, k - 1]

                            # phi 1/4 pi
                            elif ((0.125 * np.pi) <= phi[i, j, k] < (0.375 * np.pi)):
                                q = img[i + 1, j + 1, k + 1]
                                r = img[i - 1, j - 1, k - 1]

                            # phi 1/2 pi
                            elif ((0.375 * np.pi) <= phi[i, j, k] < (0.625 * np.pi)):
                                q = img[i, j + 1, k + 1]
                                r = img[i, j - 1, k - 1]

                            # phi 3/4 pi
                            elif ((0.625 * np.pi) <= phi[i, j, k] < (0.875 * np.pi)):
                                q = img[i - 1, j + 1, k + 1]
                                r = img[i + 1, j - 1, k - 1]

                        # theta 1/2 pi
                        elif ((0.375 * np.pi) <= theta[i, j, k] < (0.625 * np.pi)):
                            # phi 0
                            if (0 <= phi[i, j, k] < (0.125 * np.pi)) or ((0.875 * np.pi) <= phi[i, j, k] <= np.pi):
                                q = img[i + 1, j, k]
                                r = img[i - 1, j, k]

                            # phi 1/4 pi
                            elif ((0.125 * np.pi) <= phi[i, j, k] < (0.375 * np.pi)):
                                q = img[i + 1, j + 1, k]
                                r = img[i - 1, j - 1, k]

                            # phi 1/2 pi
                            elif ((0.375 * np.pi) <= phi[i, j, k] < (0.625 * np.pi)):
                                q = img[i, j + 1, k]
                                r = img[i, j - 1, k]

                            # phi 3/4 pi
                            elif ((0.625 * np.pi) <= phi[i, j, k] < (0.875 * np.pi)):
                                q = img[i - 1, j + 1, k]
                                r = img[i + 1, j - 1, k]

                        # theta 3/4 pi
                        elif ((0.625 * np.pi) <= theta[i, j, k] < (0.875 * np.pi)):
                            # phi 0
                            if (0 <= phi[i, j, k] < (0.125 * np.pi)) or ((0.875 * np.pi) <= phi[i, j, k] <= np.pi):
                                q = img[i + 1, j, k - 1]
                                r = img[i - 1, j, k + 1]

                            # phi 1/4 pi
                            elif ((0.125 * np.pi) <= phi[i, j, k] < (0.375 * np.pi)):
                                q = img[i + 1, j + 1, k - 1]
                                r = img[i - 1, j - 1, k + 1]

                            # phi 1/2 pi
                            elif ((0.375 * np.pi) <= phi[i, j, k] < (0.625 * np.pi)):
                                q = img[i, j + 1, k - 1]
                                r = img[i, j - 1, k + 1]

                            # phi 3/4 pi
                            elif ((0.625 * np.pi) <= phi[i, j, k] < (0.875 * np.pi)):
                                q = img[i - 1, j + 1, k - 1]
                                r = img[i + 1, j - 1, k + 1]

                        if (img[i, j, k] >= q) and (img[i, j, k] >= r):
                            Z[i, j, k] = img[i, j, k]
                        else:
                            Z[i, j, k] = 0

                    except IndexError as e:
                        pass

        return Z

    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold;
        lowThreshold = highThreshold * self.lowThreshold;

        M, N, O = img.shape
        res = np.zeros((M, N, O), dtype=np.int32)

        weak = np.int32(self.weak_voxel)
        strong = np.int32(self.strong_voxel)

        strong_i, strong_j, strong_k = np.where(img >= highThreshold)
        zeros_i, zeros_j, zeros_k = np.where(img < lowThreshold)

        weak_i, weak_j, weak_k = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j, strong_k] = strong
        res[weak_i, weak_j, weak_k] = weak

        return res

    def hysteresis(self, img):

        M, N, O = img.shape
        weak = self.weak_voxel
        strong = self.strong_voxel

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                for k in range(1, O - 1):
                    if (img[i, j, k] == weak):
                        try:
                            if ((img[i, j, k - 1] == strong) or (img[i, j, k + 1] == strong)
                                    or (img[i, j - 1, k] == strong) or (img[i, j - 1, k - 1] == strong) or (
                                            img[i, j - 1, k + 1] == strong)
                                    or (img[i, j + 1, k] == strong) or (img[i, j + 1, k - 1] == strong) or (
                                            img[i, j + 1, k + 1] == strong)
                                    or (img[i - 1, j, k] == strong) or (img[i - 1, j, k - 1] == strong) or (
                                            img[i - 1, j, k + 1] == strong)
                                    or (img[i - 1, j - 1, k] == strong) or (img[i - 1, j - 1, k - 1] == strong) or (
                                            img[i - 1, j - 1, k + 1] == strong)
                                    or (img[i - 1, j + 1, k] == strong) or (img[i - 1, j + 1, k - 1] == strong) or (
                                            img[i - 1, j + 1, k + 1] == strong)
                                    or (img[i + 1, j, k] == strong) or (img[i + 1, j, k - 1] == strong) or (
                                            img[i + 1, j, k + 1] == strong)
                                    or (img[i + 1, j - 1, k] == strong) or (img[i + 1, j - 1, k - 1] == strong) or (
                                            img[i + 1, j - 1, k + 1] == strong)
                                    or (img[i + 1, j + 1, k] == strong) or (img[i + 1, j + 1, k - 1] == strong) or (
                                            img[i + 1, j + 1, k + 1] == strong)):
                                img[i, j, k] = strong
                            else:
                                img[i, j, k] = 0
                        except IndexError as e:
                            pass
        return img

    def detect(self):
        self.img_smoothed = gaussian_filter(self.img, self.sigma)
        self.gradientMat, self.phiMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
        self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.phiMat, self.thetaMat)
        self.thresholdImg = self.threshold(self.nonMaxImg)
        self.img_edges = self.hysteresis(self.thresholdImg)

        return self.img_edges

