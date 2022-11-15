#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on 2022. 09. 05. 17:32
@author  :  A. Tóth, Sz. Szeghalmy

Jellemzőkinyerésre, illetve a szűrők, képek megjelenítésére szolgáló eszközök kaptak itt helyet.
A create_feature_map függvény egy rgb és egy depth képből előállít egy (rgb.shape[0], rgb.shape[1], n)
méretű jellemzőtérképet, ahol n a jellemzők száma. Az alapértelmezett szűrőbankkal 36.

"""

import math

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import cv2

from skimage.filters import gabor_kernel
from skimage.color import rgb2lab
from scipy import ndimage as ndi

def preprocess(image):
    image = cv2.GaussianBlur(image, ksize = (5, 5), sigmaX=0.6, sigmaY=0.6)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)



def convolve_with_fbank(image, kernels):
    """

    :param image:
    :param kernels:
    :return:
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    features = []
    for k, kernel in enumerate(kernels):
        for chs in range(3):
            features.append(lab[:,:,chs])
            filtered = cv2.filter2D(src=lab[:,:,chs], ddepth=-1, kernel=kernel)
            features.append(filtered)

    return features


def create_feature_maps(filtered, depth):
    """

    :param filtered:
    :param depth:
    :return:
    """
    h, w = filtered[0].shape[0:2]
    features = np.zeros((h, w, len(filtered)+4), dtype=np.double)
    for k, kernel in enumerate(filtered):
        features[:,:,k] = filtered[k]
    features[:,:,features.shape[2]-1] = depth
    return features

def create_filter_banks(num_angles = 4, sigmas = (1.0, 2.0, 3.0), frequencies = (0.05, 0.25, 0.5)):
    """

    :param num_angles:
    :param sigmas:
    :param frequencies:
    :return:
    """

    kernels = []
    for theta in range(num_angles):
        theta = theta / num_angles * np.pi
        for sigma in sigmas:
            for frequency in frequencies:
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels


def create_feature_map(img, depth, kernels=None):
    if kernels is None:
        kernels = create_filter_banks(4, (1.0, 2.0, 3.0), (0.05, 0.25, 0.5))
    filtered = convolve_with_fbank(img, kernels)
    feature_map = create_feature_maps(filtered, depth)
    return feature_map


def show_img_series(kernels):
    ncols = 6
    nrows = math.ceil(len(kernels) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows, ncols))
    for idx, kernel in enumerate(kernels):
        y = idx // ncols
        x = idx %  ncols
        axes[y, x].imshow(np.real(kernel))
        # axes[y, x].set_ylabel(label, fontsize=7)
        axes[y, x].set_xticks([])
        axes[y, x].set_yticks([])

    plt.show()




if __name__ == "__main__":

    kernels = create_filter_banks(4, (1.0, 2.0, 3.0), (0.05, 0.25, 0.5))
    show_img_series(kernels)

    img = cv2.imread("nyu1.png", cv2.IMREAD_COLOR)
    depth = cv2.imread("nyu2.png", cv2.IMREAD_GRAYSCALE)

    filtered = convolve_with_fbank(img, kernels)
    show_img_series(filtered)

    feature_imgs = create_feature_maps(filtered, depth)

    print(feature_imgs.shape)
    #rshaped =  features.reshape((h * w, len(filtered + 1)))
