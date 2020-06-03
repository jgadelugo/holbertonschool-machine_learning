#!/usr/bin/env python3
"""makes prediction using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes prediction using a neural network
    @network: the network model to make the prediction with
    @data: input data to make the prediction
    @verbose: boolean to print or not
    Return: the prediction for the data
    """
    return network.predict(data, verbose=verbose)
