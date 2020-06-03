#!/usr/bin/env python3
"""tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network
    @network: network model to test
    @data: input data to test the model with
    @labels: the correct one-hot labels of data
    @verbose: boolean that determines if output should
    be printed during testing
    Return: loss and accuracy of the model with the testing data
    """
    return network.evaluate(data, labels, verbose=verbose)
