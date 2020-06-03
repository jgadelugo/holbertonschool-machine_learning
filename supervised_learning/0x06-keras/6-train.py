#!/usr/bin/env python3
""" train a model using mini-bath gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """train a model using mini-bath gradient descent
    @network: model to train
    @data: np.ndarray - shape (m, classes) - of input data
    @labels: one-hot np.ndarray - shape(m, classes)
    containing the labels of data
    @batch_size: the size of the batch
    @epochs: the number of passes through data
    @validation_data: the data to validate the model with
    @early_stopping: a boolean that indicates whether early stopping
    should be used
        early stopping should only be performed if validation_data exists
        early stopping should be based on validation loss
    @patience: patience used for early stopping
    @verbose: a boolean that determines if output should be
    printed during training
    @shuffle: a boolean that determines wheher to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle,
    but for reproducibility, we have chosen to set the default to False
    Return: History object generated after training the model
    """
    if validation_data and early_stopping:
        callback = [K.callbacks.EarlyStopping(patience=patience)]
    else:
        callback = None
    return network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callback)
