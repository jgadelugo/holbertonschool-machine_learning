#!/usr/bin/env python3
"""Function randomly changes the brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """randomly changes the brightness of an image
    @image: image
    @max_delta: maximum amount the image should be brightened(or darkened)
    Return: brightened image
    """
    return tf.image.adjust_brightness(image, max_delta)
