#!/usr/bin/env python3
""" train a model using mini-bath gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """train a model using mini-bath gradient descent
    @network: model to train
    @data: np.ndarray - shape (m, classes) - of input data
    @labels: one-hot np.ndarray - shape(m, classes)
    containing the labels of data
    @batch_size: the size of the batch
    @epochs: the number of passes through data
    @verbose: a boolean that determines if output should be
    printed during training
    @shuffle: a boolean that determines wheher to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle,
    but for reproducibility, we have chosen to set the default to False
    Return: History object generated after training the model
    """
    return network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle)
