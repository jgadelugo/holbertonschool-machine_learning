#!/usr/bin/env python3
""" performs pooling on image
with custom padding"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on image
    @images: np.ndarray - shape(m, h, w) - containing multiple
    grayscale images
        @m: number of images
        @h: height in pixels of images
        @w: width in pixels of images
        @c: number of chnnels in the img
    @kernel_shape: tuple - shape(kh, kw) - containing the kernel
    for the pooling
        @kh: height of the kernel
        @kw: width of the kernel
    @stride: tuple (sh, sw)
        @sh: stride for the height of the image
        @sw: stride for the width of the image
    @mode: type of pooling
        if 'max' use max pooling
        if 'avg' use average pooling
    *only allowed to use two for loop
    Return: np.ndarray conatining the pooled images
    """
    # image variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    # kernel variables
    kh = kernel_shape[0]
    kw = kernel_shape[1]

    # stride variables
    sh = stride[0]
    sw = stride[1]

    # set height and width
    height = int((h - kh) / sh + 1)
    width = int((w - kw) / sw + 1)

    # convolution image
    new_img = np.zeros((m, height, width, c))
    image = np.arange(m)

    for i in range(height):
        for j in range(width):
            img = images[image, i * sh: i * sh + kh,
                         j * sw: j * sw + kw]
            if mode == 'avg':
                new_img[image, i, j] = np.mean(img, axis=(1, 2))
            if mode == 'max':
                new_img[image, i, j] = np.max(img, axis=(1, 2))
    return new_img
