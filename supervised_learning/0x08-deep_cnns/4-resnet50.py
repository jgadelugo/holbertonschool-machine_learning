#!/usr/bin/env python3
""" builds the ResNet-50 architecture as described in
https://arxiv.org/pdf/1512.03385.pdf"""
import tensorflow.keras as K
projection_block = __import__('3-projection_block').projection_block
identity_block = __import__('2-identity_block').identity_block


def resnet50():
    """ builds the ResNet-50 architecture as described in
    https://arxiv.org/pdf/1512.03385.pdf
    Assume input data shape (224, 224, 3)
    All convolutions inside and outside the blocks should be
    followed by batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    Return:  the keras resnet-50 model
    """
    _input = K.Input(shape=(224, 224, 3))

    # Conv - filter=64, kernel_size=(7, 7), strides=(2, 2)
    output = K.layers.Conv2D(64, 7, 2,
                             padding='same', kernel_initializer='he_normal'
                             )(_input)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.Activation('relu')(output)

    # MaxPool - pool_size=(3, 3), strides=(2, 2)
    output = K.layers.MaxPool2D(3, 2, padding='same')(output)

    output = projection_block(output, [64, 64, 256], 1)
    output = identity_block(output, [64, 64, 256])
    output = identity_block(output, [64, 64, 256])

    output = projection_block(output, [128, 128, 512], 2)
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])

    output = projection_block(output, [256, 256, 1024], 2)
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])

    output = projection_block(output, [512, 512, 2048], 2)
    output = identity_block(output, [512, 512, 2048])
    output = identity_block(output, [512, 512, 2048])

    # avg pooling pool_size=(7x7)
    output = K.layers.AvgPool2D(7)(output)
    # units=1000
    output = K.layers.Dense(1000, kernel_initializer='he_normal',
                            activation='softmax')(output)
    # return ResNet-50 Model
    return K.Model(_input, output)
