#!/usr/bin/env python3
"""update variable using the gradient descent w/ momentum optimization algo"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """training op for neural network in tensorflow using
    gradient descent with momentum optimization algorithm
    @loss: loss of the network
    @alpha: learning rate
    @beta1: momentum weight
    Return: the momentum optimization operation
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
