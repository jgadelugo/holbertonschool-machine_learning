#!/usr/bin/env python3
"""create placeholder"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """ Function returns two placeholders, x and y, for the neural network """
    x = tf.placeholder(tf.float32, [None, nx], name="x")
    y = tf.placeholder(tf.float32, [None, classes], name="y")
    return (x, y)
