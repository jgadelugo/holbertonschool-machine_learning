#!/usr/bin/env python3
"""Complete following code
Concat coinbase and bitstamp to 1417411920
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# YOUR CODE HERE
df1.index = df1['Timestamp']
df2.index = df2['Timestamp']
df = pd.concat([df2.loc[:"1417411920"], df1], keys=["bitstamp", "coinbase"])

del df['Timestamp']

print(df)
