#!/usr/bin/env python3
"""Function changes the hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """changes the hue of an image
    @image: image
    @delta: amount the hue should change
    Return: altered image
    """
    return tf.image.adjust_brightness(image, delta)
