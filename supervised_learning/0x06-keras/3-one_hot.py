#!/usr/bin/env python3
""" Converts a label vector into a one-hot matrix """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Converts a label vector into a one-hot matrix
    the last dimension of the one-hot matrix must be the number of classes
    Returns: one-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
