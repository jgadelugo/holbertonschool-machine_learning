#!/usr/bin/env python3
""" Update a variable in place using the adam optimization algorithm """


def update_variables_Adam(alpha, b1, b2, epsilon, var, grad, v, s, t):
    """ Update a variable in place using the adam optimization algorithm
    @alpha: learning rate
    @b1: weight used for the first moment
    @b2: weight used for the second moment
    @epsilon: small number to avoid division by zero
    @var: numpy.ndarray - variable to be updated
    @grad: numpy.ndarray - gradient of var
    @v: previous first moment of var
    @s: previous second moment of var
    @t: time step used for bias correction
    Return: updated variables, (first moment, second moment)
    """

    v = (b1 * v) + ((1 - b1) * grad)
    s = (b2 * s) + ((1 - b2) * grad ** 2)

    v_correct = v / (1 - (b1 ** t))
    s_correct = s / (1 - (b2 ** t))

    var -= (alpha * (v_correct / ((s_correct ** (1/2)) + epsilon)))
    return (var, v, s)
