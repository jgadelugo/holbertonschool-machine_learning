#!/usr/bin/env python3
"""Function that randomly shears an image"""
import tensorflow as tf


def shear_image(image, intensity):
    """randomly shears an image
    @image: image to rotate
    @intensity: intensity with which the image should be sheared
    Return: sheared image
    """
    imgage = tf.keras.preprocessing.image.img_to_array(image)
    shear = tf.keras.preprocessing.image.random_shear
    sheared = shear(imgage, intensity, row_axis=0, col_axis=1, channel_axis=2)
    return tf.keras.preprocessing.image.array_to_img(sheared)
