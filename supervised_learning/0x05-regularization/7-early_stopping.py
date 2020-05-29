#!/usr/bin/env python3
""" determines if you should stop gradient descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if you should stop gradient descent early
        Early stopping should occur when the validation cost of
    the network has not decreased relative to the optimal
    validation cost by more than the threshold over a specific
    patience count
    @cost: current validation cost of the neural network
    @opt_cost: lowest recorded validation cost of the neural network
    @threshold: threshold used for early stopping
    @patience: patience count used for early stopping
    @count: count of how long the threshold has not been met
    Return: a boolean of whether the network should be stopped early,
    followed by the updated count
    """
    differences = opt_cost - cost
    if differences > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return (True, count)
    return(False, count)
