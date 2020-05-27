#!/usr/bin/env python3
"""Calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix
    @confusion: confusion np.ndarray - shape (classes, classes) -
    rows represent correct labels, columns represent predicted labels
        @classes: number of classes
    Return: np.ndarray - shape (classes,) - sensitivity of each class
    """
    return confusion.diagonal() / confusion.sum(axis=1)
