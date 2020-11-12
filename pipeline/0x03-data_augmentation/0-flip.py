#!/usr/bin/env python3
"""Function that flips an image horizontally"""
import tensorflow as tf


def flip_image(image):
    """ Flips image horizontally
    @image: image to flip
    Return: flipped image
    """
    return tf.image.flip_left_right(image)
