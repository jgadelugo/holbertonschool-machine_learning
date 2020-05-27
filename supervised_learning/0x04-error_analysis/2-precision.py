#!/usr/bin/env python3
""" calculates the precision for each class in a confusion matrix:"""
import numpy as np


def precision(confusion):
    """ calculates the precision for each class in a confusion matrix:
    @confusion: confusion np.ndarray - shape (classes, classes) -
    rows represent correct labels, columns represent predicted labels
        @classes: number of classes
    Return: np.ndarray - shape (classes,) - precision of each class
    """
    return confusion.diagonal() / confusion.sum(axis=0)
