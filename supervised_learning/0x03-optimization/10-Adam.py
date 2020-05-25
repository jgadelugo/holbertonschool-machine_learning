#!/usr/bin/env python3
""" Update a variable in place using the adam optimization algorithm """
import tensorflow as tf


def create_Adam_op(loss, alpha, b1, b2, epsilon):
    """ Update a variable in place using the adam optimization algorithm
    @alpha: learning rate
    @b1: weight used for the first moment
    @b2: weight used for the second moment
    @epsilon: small number to avoid division by zero
    Return: the Adam optimization operation
    """
    return tf.train.AdamOptimizer(alpha, b1, b2, epsilon).minimize(loss)
