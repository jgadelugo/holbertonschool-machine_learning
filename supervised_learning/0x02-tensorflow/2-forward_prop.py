#!/usr/bin/env python3
"""predicts the network in tensor form"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    @x: is the placeholder for the input data
    @layer_sizes: list conatining the number of nodes
        in each layer of the network
    @activations: list of activation functions
    Returns: the prediction of the network in tensor form
    """
    i = 0
    size = len(layer_sizes)
    y = x
    while i < size:
        y = create_layer(y, layer_sizes[i], activations[i])
        i += 1
    return y
