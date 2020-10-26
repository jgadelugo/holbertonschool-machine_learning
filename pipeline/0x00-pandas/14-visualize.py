#!/usr/bin/env python3
"""Complete following code
Plot the data from 2017 and beyond at daily intervals
The column Weighted_Price should be removed
Rename the column Timestamp to Date
Convert the timestamp values to date values
Index the data frame on Date
Missing values in High, Low, Open, and Close should be set to the previous
row’s Close value
Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
# replace NaN with datapoints
df = df.rename(columns={"Timestamp": "Date"})
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df.index = df['Date']
del df['Date']


del df['Weighted_Price']

df['Close'].fillna(method='bfill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

df = df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low':'min',
                             'Close':'last', 'Volume_(BTC)':'sum',
                             'Volume_(Currency)':'sum'})

print(df)
df['2017-01-01':].plot()
plt.show()
