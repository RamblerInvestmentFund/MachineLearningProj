import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import talib as ta
import sklearn
from sklearn.model_selection import train_test_split

df = yf.download('AAPL', start='2018-01-01', end='2020-10-29')

# Technicals Dataframe
df['Percent Return'] = df['Adj Close'].pct_change() * 100
df['Daily Range'] = df['High'] - df['Low']
df['Simple MA'] = ta.SMA(df['Adj Close'], timeperiod=10)
df['Daily Close - SMA Difference'] = df['Adj Close'] - df['Simple MA']
# df['MACD'] = ta.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['RSI'] = ta.RSI(df['Adj Close'], timeperiod=9)
# df['Daily CLose - RSI'] = df['Adj Close'] - df['RSI']
df['Up Band'], df['Middle Band'], df['Down Band'] = ta.BBANDS(df['Adj Close'], timeperiod=10)
df['Daily Close - Up Band'] = abs(df['Adj Close'] - df['Up Band'])
df['Daily Close - Middle Band'] = abs(df['Adj Close'] - df['Middle Band'])
df['Daily Close - Down Band'] = abs(df['Adj Close'] - df['Down Band'])
df['Daily Close - High'] = abs(df['Adj Close'] - df['High'])
df['Daily Close - Low'] = abs(df['Adj Close'] - df['Low'])


def signal(x):
    if x['Percent Return'] > 3:
        return 1
    elif x['Percent Return'] < 3:
        return 0


df['Signal'] = df.apply(signal, axis=1)

print(df.head())

# Plotting Bands and SMA
# df[['Adj Close', 'Up Band', 'Middle Band', 'Down Band', 'Simple MA']].plot(figsize=(12,10))
# plt.show()

# Drop Unwanted Columns & creating arrays for SVM
X = np.array(df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Simple SMA', 'Up Band',
                      'Middle Band', 'Down Band', 'Signal'], 1))
Y = np.array(df['Signal'])

# print(X)
# print(Y)

# Splitting the data into training and testing data
X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y, test_size=0.15)
X2_train, X1_dev, y2_train, y1_dev = train_test_split(X1_train, y1_train, test_size=0.15)


