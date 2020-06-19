#!/usr/bin/env python3
""" builds a dense block as described in
https://arxiv.org/pdf/1608.06993.pdf"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds a dense block as described in
    https://arxiv.org/pdf/1608.06993.pdf
    @X: output from the previous layer
    @np_filters: an integer representation the number of
    filters in X
    @growth_rate: growth rate for the dense block
    @layers: number of layers in the dense block
    * use bottleneck layers used for DenseNet-B
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU)
    All weights should use he normal initialization
    Return:  concatenated output of each layer within the Dense
    Block and the number of filters within the concatenated outputs
    """
    for i in range(layers):
        layer = K.layers.BatchNormalization()(X)
        layer = K.layers.Activation('relu')(layer)

        # 1x1 convolution, filters=4*growth_rate
        layer = K.layers.Conv2D(growth_rate*4, 1,
                                kernel_initializer='he_normal')(layer)

        layer = K.layers.BatchNormalization()(layer)
        layer = K.layers.Activation('relu')(layer)
        # 3x3 convolution, filters=growth_rate
        layer = K.layers.Conv2D(growth_rate, 3, padding='same',
                                kernel_initializer='he_normal')(layer)
        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate
    return X, nb_filters
