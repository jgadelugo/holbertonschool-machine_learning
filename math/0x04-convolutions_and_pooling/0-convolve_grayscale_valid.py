#!/usr/bin/env python3
""" performs a valid convolution on grayscale image"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ performs a valid convolution on grayscale image
    @images: np.ndarray - shape(m, h, w) - containing multiple
    grayscale images
        @m: number of images
        @h: height in pixels of images
        @w: width in pixels of images
    @kernel: np.ndarray - shape(kh, kw) - containing the kernel
    for the convolution
        @kh: height of the kernel
        @kw: width of the kernel
    *only allowed to use two for loop
    Return: np.ndarray conatining the convolved images
    """
    # image variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    # kernel variables
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    height = int(h - kh + 1)
    width = int(w - kw + 1)

    new_img = np.zeros((m, height, width))

    for i in range(width):
        for j in range(height):
            img = images[:, j: j + kh, i: i + kw]
            new_img[:, j, i] = np.sum(kernel * img, axis=(1, 2))
    return new_img
