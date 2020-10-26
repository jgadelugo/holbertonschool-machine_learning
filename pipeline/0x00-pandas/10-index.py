#!/usr/bin/env python3
"""Complete following code
replace NaN with datapoints
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.reset_index(inplace=True, drop=True)
df.index = df['Timestamp']

del df['Timestamp']

print(df.tail())
