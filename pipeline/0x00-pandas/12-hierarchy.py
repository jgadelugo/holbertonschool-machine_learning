#!/usr/bin/env python3
"""Complete following code
Concatenate th bitstamp and coinbase tables
from timestamps 1417411980 to 1417417980
Display the rows in chronological order
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# YOUR CODE HERE
df1.index = df1['Timestamp']
df2.index = df2['Timestamp']
df1 = df1.loc["1417411980":"1417417980"]
df2 = df2.loc["1417411980":"1417417980"]

df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

del df['Timestamp']

df = df.sort_index()

print(df)
