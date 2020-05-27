#!/usr/bin/env python3
"""calculates the F1 score of a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix
    @confusion: confusion np.ndarray - shape (classes, classes) -
    rows represent correct labels, columns represent predicted labels
        @classes: number of classes
    Return: np.ndarray - shape (classes,) - F1 score of each class
    """
    p = precision(confusion)
    s = sensitivity(confusion)
    return 2 * p * s / (p + s)
