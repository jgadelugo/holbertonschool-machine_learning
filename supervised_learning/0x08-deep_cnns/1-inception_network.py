#!/usr/bin/env python3
""" builds the inception network as described in
https://arxiv.org/pdf/1409.4842.pdf"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds the inception network as described in
    https://arxiv.org/pdf/1409.4842.pdf
    Assume input data shape (224, 224, 3)
    Use ReLU inside and outside the inception block
    Return: keras Googlenet model
    """
    _input = K.Input(shape=(224, 224, 3))

    # Conv - filter=64, kernel_size=(7, 7), strides=(2, 2)
    output = K.layers.Conv2D(64, 7, 2, activation='relu',
                             padding='same', input_shape=(224, 224, 3)
                             )(_input)
    # MaxPool - pool_size=(3, 3), strides=(2, 2)
    output = K.layers.MaxPool2D(3, 2, padding='same')(output)

    # Conv filter=64, kernel_size=(1, 1), strides=(1, 1)
    output = K.layers.Conv2D(64, 1, 1, activation='relu',
                             padding='same')(output)
    # Conv - filter=192, kernel_size=(3, 3), strides=(1, 1)
    output = K.layers.Conv2D(192, 3, 1, activation='relu',
                             padding='same')(output)
    # MaxPool - pool_size=(3, 3), strides=(2, 2)
    output = K.layers.MaxPool2D(3, 2, padding='same')(output)

    # inception block
    output = inception_block(output, [64, 96, 128, 16, 32, 32])
    output = inception_block(output, [128, 128, 192, 32, 96, 64])
    # MaxPool - pool_size=(3, 3), strides=(2, 2)
    output = K.layers.MaxPool2D(3, 2, padding='same')(output)
    # inception block
    output = inception_block(output, [192, 96, 208, 16, 48, 64])
    output = inception_block(output, [160, 112, 224, 24, 64, 64])
    output = inception_block(output, [128, 128, 256, 24, 64, 64])
    output = inception_block(output, [112, 144, 288, 32, 64, 64])
    output = inception_block(output, [256, 160, 320, 32, 128, 128])
    # MaxPool - pool_size=(3, 3), strides=(2, 2)
    output = K.layers.MaxPool2D(3, 2, padding='same')(output)
    # inception block
    output = inception_block(output, [256, 160, 320, 32, 128, 128])
    output = inception_block(output, [384, 192, 384, 48, 128, 128])
    # Avg pool - pool_size=(7, 7)
    output = K.layers.AvgPool2D(7, 1)(output)
    # Dropout rate=.4
    output = K.layers.Dropout(.4)(output)

    init = K.initializers.he_normal(seed=None)
    # units=1000, softmax
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(output)

    # return GoogleNet Model
    return K.Model(_input, output)
