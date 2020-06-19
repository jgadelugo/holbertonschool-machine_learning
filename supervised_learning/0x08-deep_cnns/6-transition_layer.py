#!/usr/bin/env python3
""" builds a transition layer as described in
https://arxiv.org/pdf/1608.06993.pdf"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ builds a transition layer as described in
    https://arxiv.org/pdf/1608.06993.pdf
    @X: output from the previous layer
    @np_filters: an integer representation the number of
    filters in X
    @compression: the compression factor for the transition layer
    * implement compression as used in DenseNet-C
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU)
    All weights should use he normal initialization
    Return:   output of the transition layer and the number
    of filters within the output
    """
    output = K.layers.BatchNormalization()(X)
    output = K.layers.Activation('relu')(output)
    output = K.layers.Conv2D(int(nb_filters * compression), 1,
                             kernel_initializer='he_normal')(output)

    # transition layer
    X = K.layers.AvgPool2D(2)(output)
    # number of filters within the output
    nb_filters = int(nb_filters * compression)
    return X, nb_filters
