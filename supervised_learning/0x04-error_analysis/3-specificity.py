#!/usr/bin/env python3
""" calculates the specificity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix
    @confusion: confusion np.ndarray - shape (classes, classes) -
    rows represent correct labels, columns represent predicted labels
        @classes: number of classes
    Return: np.ndarray - shape (classes,) - specificity of each class
    """
    true_pos = confusion.diagonal()

    false_neg = confusion.sum(axis=1) - true_pos
    false_pos = confusion.sum(axis=0) - true_pos

    true_neg = confusion.sum() - (true_pos + false_neg + false_pos)

    return true_neg / (true_neg + false_pos)
