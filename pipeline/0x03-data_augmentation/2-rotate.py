#!/usr/bin/env python3
"""Function that rotates an image by 90 degrees counter-clockwise"""
import tensorflow as tf


def rotate_image(image):
    """rotates an image by 90 degrees counter-clockwise
    @image: image to rotate
    Return: rotated image
    """
    return tf.image.rot90(image=image, k=1, name=None)
