#!/usr/bin/env python3
"""that builds a modified version of the LeNet-5 architecture using keras"""
import tensorflow.keras as K


def lenet5(X):
    """that builds a modified version of the LeNet-5 architecture using keras
    @X: K.Input - shape(m, 28, 28, 1) input images for the network
        @m: number of images
    * Model Layers in order
        @lay1: convolutional layer with 6 kernels shape 5x5 with same padding
        @lay2: Max pooling layer with kernels of shape 2x2 with 2x2 strides
        @lay3: Convolutional layer with 16 kernels shape 5x5 with valid padding
        @lay4: Max pooling layer with kernels of shape 2x2 with 2x2 strides
        @lay5: Fully connected layer with 120 nodes
        @lay6: Fully connected layer with 84 nodes
        @lay7: Fully connected softmax output layer with 10 nodes
    * all layers requiring initialization - initialize kernels with he_normal
    * all hidden layers requiring activation should use relu
    Returns: a tensor for the softmax activated output
             a training operation that utilizes Adam optimization
             a tensor for the loss of the network
             a tensor for the accuracy of the network
    """
    init = K.initializers.he_normal(seed=None)

    lay1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                           activation='relu', kernel_initializer=init)(X)

    lay2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(lay1)

    lay3 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                           activation='relu', kernel_initializer=init)(lay2)

    lay4 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(lay3)

    # flatten before dense layers
    lay4 = K.layers.Flatten()(lay4)

    lay5 = K.layers.Dense(units=120, activation='relu',
                          kernel_initializer=init)(lay4)

    lay6 = K.layers.Dense(units=84, activation='relu',
                          kernel_initializer=init)(lay5)

    # output
    lay7 = K.layers.Dense(units=10,
                          activation='softmax',
                          kernel_initializer=init)(lay6)

    model = K.models.Model(X, lay7)

    model.complile(optimizer=K.optimizers.Adam(),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    return model
