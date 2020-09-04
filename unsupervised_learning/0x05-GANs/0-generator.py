#!/usr/bin/env python3
"""Creates a simple generator network for MNIST digits"""
import tensorflow as tf


def generator(Z):
    """Creates a simple generator network for MNIST digits
    @Z: tf.tensor containing the input to the generator network
    * network should have two layers
        *the first layer should have 128 nodes and use relu activation with
        name layer_1
        *the second layer should have 784 nodes and use a sigmoid activation
        with name layer_2
    * all variables in the network should have the scope generator with
    with reuse=tf.AUTO_REUSE
    Return: X
        @X: tf.tensor containing the generated image
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.Dense(input=Z, units=128, name='layer_1',
                                  activation='relu')
        X = tf.layers.Dense(input=layer_1, units=784, name='layer_2',
                            activation='sigmoid')
    return X
