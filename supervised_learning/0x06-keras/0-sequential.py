#!/usr/bin/env python3
"""Funciton that builds neural network with keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lamb, keep_prob):
    """builds a neural network with keras
    @nx: number of input features to the network
    @layers: list of the number of nodes in each layer
    @activations: list of activation functions used for each layer
    @lamb: L2 regularization parameter
    @keep_prob: probability that a node will be kept for dropout
    Return: keras model
    """
    # create a regularizer that applies an L2 regularization penalty

    lay = K.layers.Dense(layers[0], input_shape=(nx,),
                         activation=activations[0],
                         kernel_regularizer=K.regularizers.l2(lamb))

    model = K.Sequential([lay])

    for layer, activ in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layer, activation=activ,
                                 kernel_regularizer=K.regularizers.l2(lamb)))
    return model
