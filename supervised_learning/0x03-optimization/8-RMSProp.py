#!/usr/bin/env python3
""" updates a variable using RMSProp optimization algorithm using tf"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, b2, epsilon):
    """ updates a variable using RMSProp optimization algorithm
    using tensorflow
    @alpha: learning rate
    @b2: RMSProp weight
    @epsilon: small number to avoid division by zero
    Return: updated variable and the new moment
    """
    return tf.train.RMSPropOptimizer(alpha, b2, epsilon=epsilon).minimize(loss)
