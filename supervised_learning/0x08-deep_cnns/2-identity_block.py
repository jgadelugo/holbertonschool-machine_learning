#!/usr/bin/env python3
""" builds an identity block as described in
https://arxiv.org/pdf/1512.03385.pdf"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ builds an identity block as described in
    https://arxiv.org/pdf/1512.03385.pdf
    @A_prev: output from the previous layer
    @filters: tuple or list containing F11, F3, F12
        @F11: number of filters in the first 1x1 convolution
        @F3: number of filters in the 3x3 convolution
        @F12: number of filters in the second 1x1 convolution
    Use ReLU inside and outside the inception block
    All waits use he normal initialization
    Return: activation output of the identity block
    """
    F11, F3, F12 = filters

    # first 1x1 convolution
    output = K.layers.Conv2D(F11, 1, padding='same',
                             kernel_initializer='he_normal')(A_prev)
    output = K.layers.BatchNormalization(axis=3)(output)
    output = K.layers.Activation('relu')(output)

    # 3x3 convolution
    output = K.layers.Conv2D(F3, 3, padding='same',
                             kernel_initializer='he_normal')(output)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.Activation('relu')(output)

    # second 1x1 convolution
    output = K.layers.Conv2D(F12, 1, kernel_initializer='he_normal'
                             )(output)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.add([output, A_prev])

    # activation output of the identity block
    return K.layers.Activation('relu')(output)
