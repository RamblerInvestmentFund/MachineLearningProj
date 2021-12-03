import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import talib as ta

df = yf.download('AAPL', start='2018-01-01', end='2020-10-29')

df['Percent Return'] = df['Adj Close'].pct_change() * 100
df['Daily Range'] = df['High'] - df['Low']
df['Simple MA'] = ta.SMA(df['Adj Close'], timeperiod=10)
df['Daily Close - SMA Difference'] = df['Adj Close'] - df['Simple MA']
# df['MACD'] = ta.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['RSI'] = ta.RSI(df['Adj Close'], timeperiod=9)
df['Daily CLose - RSI'] = df['Adj Close'] - df['RSI']
df['Up Band'], df['Middle Band'], df['Down Band'] = ta.BBANDS(df['Adj Close'], timeperiod=10)
df['Daily Close - Up Band'] = df['Adj Close'] - df['Up Band']
df['Daily Close - Middle Band'] = df['Adj Close'] - df['Middle Band']
df['Daily Close - Down Band'] = df['Adj Close'] - df['Down Band']

def signal(x):
    if x['Percent Return'] > 3:
        return 1
    elif x['Percent Return'] < 3:
        return 0

df['Signal'] = df.apply(signal, axis=1)


print(df.head())


# df['Close'].plot()
# df['Percent Return'].plot()



