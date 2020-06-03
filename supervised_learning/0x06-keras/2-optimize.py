
#!/usr/bin/env python3
""" Set up Adam optimization for keras model
with categorical crossentropy loss and accuracy metrics"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Set up Adam optimization for keras model
    with categorical crossentropy loss and accuracy metrics
    @network: model to optimize
    @alpha: learning rate
    @beta1: first adam optimization parameter
    @beta2: second adam optimization parameter
    Return: None
    """
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
