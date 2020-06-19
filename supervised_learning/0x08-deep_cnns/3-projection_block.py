#!/usr/bin/env python3
""" builds a projection block as described in
https://arxiv.org/pdf/1512.03385.pdf"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ builds a projection block as described in
    https://arxiv.org/pdf/1512.03385.pdf
    @A_prev: output from the previous layer
    @filters: tuple or list containing F11, F3, F12
        @F11: number of filters in the first 1x1 convolution
        @F3: number of filters in the 3x3 convolution
        @F12: number of filters in the second 1x1 convolution as
        well as the 1x1 convolution in the shortcut connection
    @s: stride of the first convolution in both the main path and
    shortcut connection
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All waits use he normal initialization
    Return: activated output of the projection block
    """
    F11, F3, F12 = filters

    # first 1x1 convolution
    output = K.layers.Conv2D(F11, 1, s,
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

    # shortcut path 1x1 convolution
    shotcut = K.layers.Conv2D(F12, 1, s,
                              kernel_initializer='he_normal')(A_prev)

    shotcut = K.layers.BatchNormalization()(shotcut)

    output = K.layers.add([output, shotcut])
    # activation output of the projection block
    return K.layers.Activation('relu')(output)
