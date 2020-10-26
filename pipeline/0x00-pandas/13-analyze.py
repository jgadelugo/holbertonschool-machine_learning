#!/usr/bin/env python3
"""Complete following code
Get descriptive statistics for all columns
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
stats = df.describe()

del stats['Timestamp']

print(stats)
