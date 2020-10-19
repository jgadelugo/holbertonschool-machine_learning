#!/usr/bin/env python3
import pandas as pd


def preprocess():
    df_price = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")

    # replace NaN with datapoints
    df_price['Close'].fillna(method='bfill', inplace=True)
    df_price['High'].fillna(df_price['Close'], inplace=True)
    df_price['Low'].fillna(df_price['Close'], inplace=True)
    df_price['Open'].fillna(df_price['Close'], inplace=True)
    df_price['Volume_(BTC)'].fillna(0, inplace=True)
    df_price['Volume_(Currency)'].fillna(0, inplace=True)
    df_price['Weighted_Price'].fillna(0, inplace=True)
    return df_price