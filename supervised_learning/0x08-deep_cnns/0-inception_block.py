#!/usr/bin/env python3
"""Build an inception block as described in
https://arxiv.org/pdf/1409.4842.pdf"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Build an inception block as described in
    https://arxiv.org/pdf/1409.4842.pdf
    @A_prev: output from the previous layer
    @filters is a tuple or list containing (F1, F3R, F3, F5R, F5, FPP)
        @F1: Number of filters in the 1x1 convolution
        @F3R: Number of filters in the 1x1 convolution before 3x3 convolution
        @F3: number of filters in the 3x3 convolution
        @F5R: number of filters in the 5x5 convolution
        @FPP: number of filters in the 1x1 convolution after max pooling
    All convolutions inside inception block should use
    rectified linear activation (Relu)
    Return: concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    layer1x1 = K.layers.Conv2D(F1, 1, activation='relu')(A_prev)

    layer3x3 = K.layers.Conv2D(F3R, 1, activation='relu')(A_prev)
    layer3x3 = K.layers.Conv2D(F3, 3, padding='same',
                               activation='relu')(layer3x3)

    layer5x5 = K.layers.Conv2D(F5R, 1, activation='relu')(A_prev)
    layer5x5 = K.layers.Conv2D(F5, 5, padding='same',
                               activation='relu')(layer5x5)

    pool = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    pool = K.layers.Conv2D(FPP, 1, activation='relu')(pool)

    return K.layers.concatenate([layer1x1, layer3x3, layer5x5, pool])
