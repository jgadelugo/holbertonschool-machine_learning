#!/usr/bin/env python3
"""Creates the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network
    @loss: loss of the networks prediction
    @alpha: learning rate
    Returns: operation that trains the network using gradient descent
    """
    optimize = tf.train.GradientDescentOptimizer(alpha)
    return optimize.minimize(loss)
