#!/usr/bin/env python3
"""builds a modified version of the LeNet-5 architecture using tensorflow"""

import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5 architecture using tensorflow
    @x: tf.placeholder - shape(m, 28, 28, 1) input images for the network
        @m: number of images
    @y: tf.placeholder - shape(m, 10) - the one-hot labels for the network
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
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    lay1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation=activation, kernel_initializer=init)(x)

    lay2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(lay1)

    lay3 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation=activation,
                            kernel_initializer=init)(lay2)

    lay4 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(lay3)

    # flatten before dense layers
    lay4 = tf.layers.Flatten()(lay4)

    lay5 = tf.layers.Dense(units=120, activation=activation,
                           kernel_initializer=init)(lay4)

    lay6 = tf.layers.Dense(units=84, activation=activation,
                           kernel_initializer=init)(lay5)

    # output
    lay7 = tf.layers.Dense(units=10, kernel_initializer=init)(lay6)

    activated_output = tf.nn.softmax(lay7)
    loss = tf.losses.softmax_cross_entropy(y, lay7)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    equal = tf.equal(tf.argmax(lay7, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return activated_output, train_op, loss, accuracy
