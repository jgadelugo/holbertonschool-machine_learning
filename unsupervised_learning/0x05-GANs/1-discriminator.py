#!/usr/bin/env python3
"""creates a discriminator network for MNIST digits"""
import tensorflow as tf
import numpy as np


def discriminator(X):
    """creates a discriminator network for MNIST digits
    @X: tf.tensor - input to the discriminator network
    * network should have two layers
        *the first layer should have 128 nodes and use relu activation with
        name layer_1
        *the second layer should have 1 node and use a sigmoid activation
        with name layer_2
    * All variables in the network should have the scope discriminator
    with reuse=tf.AUTO_REUSE
    Return: Y
        @Y: tf.tensor - classification made by the discriminator
    """
    Y = None

    return Y
