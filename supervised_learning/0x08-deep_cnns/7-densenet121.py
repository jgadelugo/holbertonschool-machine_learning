#!/usr/bin/env python3
"""builds the DenseNet-121 architecture as described in
https://arxiv.org/pdf/1608.06993.pdf"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """builds the DenseNet-121 architecture as described in
    https://arxiv.org/pdf/1608.06993.pdf
    @growth_rate: growth rate
    @compression: the compression factor
    *assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU)
    All weights should use he normal initialization
    Return: keras model of the DenseNet-121
    """
    X = K.Input((224, 224, 3))

    output = K.layers.BatchNormalization()(X)
    output = K.layers.Activation('relu')(output)
    # 7x7 convolution, filters=2*growth_rate strides=(2, 2)
    output = K.layers.Conv2D(2*growth_rate, 7, 2, padding='same',
                             kernel_initializer='he_normal')(output)
    # 3x3 max pool, stride=(2,2)
    output = K.layers.MaxPool2D(2)(output)

    # Dense block
    output, nb_filters = dense_block(output, 2*growth_rate, growth_rate, 6)

    output, nb_filters = transition_layer(output, nb_filters, compression)
    output, nb_filters = dense_block(output, nb_filters, growth_rate, 12)

    output, nb_filters = transition_layer(output, nb_filters, compression)
    output, nb_filters = dense_block(output, nb_filters, growth_rate, 24)

    output, nb_filters = transition_layer(output, nb_filters, compression)
    output, nb_filters = dense_block(output, nb_filters, growth_rate, 16)

    # 7x7 Avg pool
    output = K.layers.AvgPool2D(7)(output)

    # units=1000, activation='softmax'
    output = K.layers.Dense(1000, kernel_initializer='he_normal',
                            activation='softmax')(output)
    # return DenseNet-121 keras model
    return K.Model(X, output)
