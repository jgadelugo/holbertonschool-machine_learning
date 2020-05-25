#!/usr/bin/env python3
"""updates the learning rate using inverse time decay """
import tensorflow as tf


def learning_rate_decay(a, d_rate, g_step, d_step):
    """ updates the learning rate using inverse time decay
    @a: learning rate
    @d_rate: weight used to determine the rate at which alpha will decay
    @g_step: number of passes of gradient descent that have elapsed
    @d_step: number of passes of gradient descent that should occur
    before alpha is decayed further
    Return: updated value for alpha
    """
    return tf.train.inverse_time_decay(a, g_step, d_step, d_rate, True)
