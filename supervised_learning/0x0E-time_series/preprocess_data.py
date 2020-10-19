#!/usr/bin/env python3
""" Clean data"""
import pandas as pd


def preprocess():
    """ Clean data"""
    df_p = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")

    # replace NaN with datapoints
    df_p['Close'].fillna(method='bfill', inplace=True)
    df_p['High'].fillna(df_p['Close'], inplace=True)
    df_p['Low'].fillna(df_p['Close'], inplace=True)
    df_p['Open'].fillna(df_p['Close'], inplace=True)
    df_p['Volume_(BTC)'].fillna(0, inplace=True)
    df_p['Volume_(Currency)'].fillna(0, inplace=True)
    df_p['Weighted_Price'].fillna(0, inplace=True)
    return df_p
