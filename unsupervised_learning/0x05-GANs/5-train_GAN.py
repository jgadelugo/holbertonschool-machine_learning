#!/usr/bin/env python3
"""trains a GAN:"""
import tensorflow as tf
import numpy as np
train_generator = __import__('2-train_generator').train_generator
train_discriminator = __import__('3-train_discriminator').train_discriminator
sample_Z = __import__('4-sample_Z').sample_Z


def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'):
    """trains a GAN:
    @X: np.ndarray shape (m, 784) containing the real data input
        @m: number of real data samples
    @epochs: number of epochs that the each network should be trained for
    @batch_size: batch size that should be used during training
    @Z_dim: number of dimensions for the randomly generated input
    @save_path: path to save the trained generator
        *Create the tf.placeholder for Z and add it to the graph's collection
    * the discriminator and generator training should be altered after one epoch
    """
    pass
