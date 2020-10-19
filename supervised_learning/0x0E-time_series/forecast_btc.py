#!/usr/bin/env python3
""" Train and show """
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# load the dataset
data = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',
                   parse_dates=[0], date_parser=dateparse)
data['Timestamp'] = data['Timestamp'].dt.tz_localize(None)
data = data.groupby([pd.Grouper(key='Timestamp',
                    freq='H')]).first().reset_index()
data = data.set_index('Timestamp')
data = data[['Close']]
data['Close'].fillna(method='bfill', inplace=True)

# split data
split_date = '25-Jun-2018'
data_train = data.loc[data.index <= split_date].copy()
data_test = data.loc[data.index > split_date].copy()

# Data preprocess
training_set = data_train.values
training_set = np.reshape(training_set, (len(training_set), 1))

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38",
             "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = data.plot(style='', figsize=(15, 5), color=color_pal[0],
              title='BTC Close Price (USD) by Hours')
