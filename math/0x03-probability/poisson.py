#!/usr/bin/env python3
""" Create a class Poisson that represents a poisson distribution"""
_e = 2.7182818285


class Poisson:
    """ represents a poisson distribution """

    def __init__(self, data=None, lambtha=1.0):
        """
        initialize - initialize poisson distribution
        @data: to estimate the distribution
        @lambtha: expected number of occurences in a given time frame
        """
        self.lambtha = lambtha
        self.data = data

    @property
    def data(self):
        """ getter for data """
        return self.__data

    @data.setter
    def data(self, value):
        """ Setter for data"""
        if value is None:
            self.__data = None
        elif not isinstance(value, list):
            raise TypeError("data must be a list")
        elif len(value) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.__data = value[:]
            self.__lambtha = sum(value) / len(value)

    @property
    def lambtha(self):
        """ getter for lambtha """
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, value):
        """ setter for lambtha"""
        if value < 1:
            raise ValueError("lambtha must be a positive value")
        self.__lambtha = float(value)

    def pmf(self, k):
        """
        calculates the value of the PMF for a given # of successes
        @k: number of successes
        """
        k = int(k)
        if k < 0:
            return 0
        x = 1
        for i in range(1, k + 1):
            x *= i
        lambtha = self.lambtha
        pmf = _e ** -lambtha * lambtha ** i / x
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        @k number of success
        """
        k = int(k)
        if k < 0:
            return 0
        lambtha = self.lambtha
        cdf = 0
        x = 1
        for i in range(k + 1):
            x *= i or 1
            cdf += _e ** -lambtha * lambtha ** i / x
        return cdf
