#!usr/bin/env/env python
""" Function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """calculates the derivative of a polynomial """
    size = len(poly)
    for x in range(size):
        poly[x] *= x
    return poly[1:]