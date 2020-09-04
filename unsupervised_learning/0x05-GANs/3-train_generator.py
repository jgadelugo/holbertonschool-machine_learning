#!/usr/bin/env python3
"""creates the loss tensor and training op for the generator"""
import tensorflow as tf
import numpy as np
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_generator(Z):
    """ creates the loss tensor and training op for the generator
    @Z: tf.tensor containing the input to the generator network
    @X: tf.tensor - input to the discriminator network
    * discriminator should minimize the negative modified minimax loss
    * discriminator should be trained using Adam optimization
    * generator should NOT be trained
    Return: loss, train_op
        @loss: discriminator loss
        @train_op: training operator for the discriminator
    """
    loss, train_op = None, None

    return loss, train_op
