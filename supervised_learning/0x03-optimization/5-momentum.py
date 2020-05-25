#!/usr/bin/env python3
"""update variable using the gradient descent w/ momentum optimization algo"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """update variable using the gradient descent w/ momentum optimization algo
    @alpha: is the learning rate
    @beta1:  is the momentum weight
    @var: numpy.ndarray containing the variable to be updated
    @grad: numpy.ndarray containing the gradient of var
    @v: previous first moment of var
    Return: the updated variable and the new moment, respectively
    """

    v = (beta1 * v) + ((1 - beta1) * grad)
    var -= (alpha * v)
    return var, v
