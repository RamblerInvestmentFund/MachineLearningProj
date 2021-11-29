import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import talib as ta

df = yf.download('AAPL', start='2018-01-01', end='2020-10-29')

df['Percent Return'] = df['Adj Close'].pct_change() * 100
df['Range'] = df['High'] - df['Low']
df['Simple MA'] = ta.SMA(df['Adj Close'], timeperiod=5)
print(df.head())


# df['Close'].plot()
# df['Percent Return'].plot()



