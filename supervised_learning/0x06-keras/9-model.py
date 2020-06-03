#!/usr/bin/env python3
"""save an entire model and load an entire model"""
import tensorflow.keras as K


def save_model(network, filename):
    """save an entire model
    @network: model to save
    @filename: path to save location
    """
    network.save(filename)


def load_model(filename):
    """load an entire model
    @filename: path to the file
    """
    return K.models.load_model(filename)
