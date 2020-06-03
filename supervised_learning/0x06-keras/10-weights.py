#!/usr/bin/env python3
"""saves a model’s weights and loads a model’s weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves a model’s weights
    @network: model whose weights should be saved
    @filename: path to save file
    @save_format: format in which the weights should be saved
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """loads a model’s weights
    @network: model to which the weight should be loaded
    @filename: path of the file"""
    return network.load_weights(filename)
