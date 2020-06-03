#!/usr/bin/env python3
"""saves a model’s configuration in JSON format and
loads a model with a specific configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves a model’s configuration in JSON format
    @network: model whose configuration should be saved
    @filename: path to save in
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """loads a model with a specific configuration
    @filename: path to file
    Return: loaded model
    """
    with open(filename) as f:
        model_s = f.read()
    return K.models.model_from_json(model_s)
