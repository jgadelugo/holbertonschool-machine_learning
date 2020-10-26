#!/usr/bin/env python3
"""Function that creates a pd.DataFrame from a csv file"""
import pandas as pd


def from_file(filename, delimiter):
    """Function that creates a pd.DataFrame from file
    @filename: file to load from
    @column separator
    Return: pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
