#!/usr/bin/env python3
"""Function that random performs crop of an image"""
import tensorflow as tf


def crop_image(image, size):
    """random crop of an image
    @image: image to flip
    @size: tuple containing the size of the crop
    Return: cropped image
    """
    return tf.random_crop(image, size=size)
