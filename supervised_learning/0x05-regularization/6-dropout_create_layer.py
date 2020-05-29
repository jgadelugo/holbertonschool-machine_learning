#!/usr/bin/env python3
"""creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer of a neural network using dropout
    @prev: tensor containing the output of the previous layer
    @n: number of nodes the new layer should contain
    @activation: the activation function that should be used on the layer
    @lambth: the probability that a node will be kept
    Return: output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.layers.Dropout(keep_prob)

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, kernel_regularizer=reg)
    return layer(prev)
