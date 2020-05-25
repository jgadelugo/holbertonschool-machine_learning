#!/usr/bin/env python3
""" updates a variable using RMSProp optimization algorithm"""


def update_variables_RMSProp(alpha, b2, epsilon, var, grad, s):
    """ updates a variable using RMSProp optimization algorithm
    @alpha: learning rate
    @b2: RMSProp weight
    @epsilon: small number to avoid division by zero
    @var: numpy.ndarray - variable to be updated - varience
    @grad: numpy.ndarray - gradient of var
    @s: previous second moment of var
    Return: updated variable and the new moment
    """
    s = b2 * s + (1 - b2) * grad ** 2
    var -= alpha * grad / ((s ** (1/2)) + epsilon)
    return (var, s)
