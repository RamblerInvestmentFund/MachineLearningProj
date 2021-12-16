import yfinance as yf
import numpy as np
import pandas as pd

import talib as ta
from talib import MA_Type

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

"create datasets from yahoo finance"
"preprocesses data for analysis "


def preprocess(ticker):
    """returns a pandas dataframe with technical indicators"""

    ## download dataset
    df = yf.download(tickers=ticker,start="2001-01-01")

    ## Technical indicators
    df["High Shifted"] = df["High"].shift(1)
    df["Low Shifted"] = df["Low"].shift(1)
    df["Close Shifted"] = df["Close"].shift(1)

    df["Upper BBand"], df["Middle BBand"], df["Lower BBand"] = ta.BBANDS(
        df["Close Shifted"], timeperiod=20
    )
    df["RSI"] = ta.RSI(np.array(df["Close Shifted"]), timeperiod=14)
    df["Macd"], df["Macd Signal"], df["Macd Hist"] = ta.MACD(
        df["Close Shifted"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["Momentum"] = ta.MOM(df["Close Shifted"], timeperiod=12)

    df["Gain Value"] = df["Open"].shift(-1) - df["Open"]
    'used in simulation not in fitting the model'

    '''
        df["Returns"] = df["Open"] / df["Open"].shift(1)
            predicts the past

        we need to predict future returns with past data
            (not yesterday's returns)
    '''

    df["Returns"] = np.log(df["Open"].shift(-1) / df["Close"])
    df["Signal"] = df["Returns"].apply(lambda x: 1 if x > 0 else 0)
    df = df[40:]
    df = df.dropna()

    return df


def split(df, ratio=0.30):
    """splits dataset df into train and test arrays"""

    ## splitting the dataset
    max_abs_scaler = preprocessing.MaxAbsScaler()

    df = df.drop(["High", "Low", "Close", "Adj Close", "Gain Value"], axis=1, inplace=False)

    X = np.array(df.drop(["Signal", "Returns"], axis=1))
    X = max_abs_scaler.fit_transform(X)
    Y = np.array(df["Signal"])

    return train_test_split(X, Y, test_size=ratio)


def load(ticker):
    """reads csv into pandas dataframe"""
    return pd.read_csv(f"./datasets/csv_{ticker}.csv")


def main():

    ticker = input("input stock ticker: ").upper()
    df = preprocess(ticker)
    df.to_csv(f"./datasets/csv_{ticker}.csv")

    # X_train, X_test, y_train, y_test = split(df)


if __name__ == "__main__":
    main()
