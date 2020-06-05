#!/usr/bin/env python3
""" performs a valid convolution on grayscale image
with custom padding"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ performs a valid convolution on grayscale image
    with custom padding with channels
    @images: np.ndarray - shape(m, h, w) - containing multiple
    grayscale images
        @m: number of images
        @h: height in pixels of images
        @w: width in pixels of images
        @c: number of chnnels in the img
    @kernel: np.ndarray - shape(kh, kw) - containing the kernel
    for the convolution
        @kh: height of the kernel
        @kw: width of the kernel
        @nc: number of kernels
    @padding: is a tuple of (ph, pw)
        *if 'same', performs a same convolution
        *if 'valid', performs a valid convolution
        @ph: padding for the height
        @pw: padding for the width
        *img should be padded with 0's
    @stride: tuple (sh, sw)
        @sh: stride for the height of the image
        @sw: stride for the width of the image
    *only allowed to use three for loop
    Return: np.ndarray conatining the convolved images
    """
    # image variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    # kernel variables
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[2]

    # stride variables
    sh = stride[0]
    sw = stride[1]

    # padding
    if isinstance(padding, tuple):
        pad_h = padding[0]
        pad_w = padding[1]
    elif padding == 'same':
        pad_h = int(((h - 1) * sh + kh - h) / 2) + 1
        pad_w = int(((w - 1) * sw + kw - w) / 2) + 1
    else:
        pad_h = 0
        pad_w = 0

    img_pad = np.pad(images,
                     pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                (0, 0)),
                     mode='constant', constant_values=0)

    # set height and width
    height = int((h + 2 * pad_h - kh) / sh + 1)
    width = int((w + 2 * pad_w - kw) / sw + 1)

    # convolution image
    new_img = np.zeros((m, height, width, nc))

    for i in range(height):
        for j in range(width):
            for z in range(nc):
                img = img_pad[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
                kernel = kernels[:, :, :, z]
                new_img[:, i, j, z] = np.sum(kernel * img, axis=(1, 2, 3))
    return new_img
