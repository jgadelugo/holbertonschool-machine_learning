#!/usr/bin/env python3
""" Update a variable in place using the adam optimization algorithm """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Update a variable in place using the adam optimization algorithm
    @alpha: learning rate
    @decay_rate: weight used to determine the rate at which alpha will decay
    @global_step: number of passes of gradient descent that have elapsed
    @decay_step: number of passes of gradient descent that should occur
    before alpha is decayed further
    Return: updated value for alpha
    """
    return alpha / (1 + (decay_rate * (global_step // decay_step)))
