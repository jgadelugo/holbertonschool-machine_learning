#!/usr/bin/env python3
""" performs a valid convolution on grayscale image"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a valid convolution on grayscale image with padding
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

    # padding
    pad_h = max(int((kh - 1) / 2), int(kh / 2))
    pad_w = max(int((kw - 1) / 2), int(kw / 2))

    img_pad = np.pad(images,
                     pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                     mode='constant', constant_values=0)

    # convolution image
    new_img = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            img = img_pad[:, i: i + kh, j: j + kw]
            new_img[:, i, j] = np.sum(kernel * img, axis=(1, 2))
    return new_img
