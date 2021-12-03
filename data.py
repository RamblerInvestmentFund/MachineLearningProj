import yfinance as yf
import pandas as pd

import talib as ta
from talib import MA_Type

'create datasets from yahoo finance'
'preprocesses data for analysis '

def fdayGain(x):
    return x["Close"] - x["Open"]

def signal(x):
    return 1 if x['Returns'] >= 0 else 0

def main():
    stocks = ["AAPL", "MSFT", "SPY", "QQQ", "DIA", "TLT", "GLD", "CVX", "KO", "PEP", "PG", "JNJ", "GSK"]

    for i in stocks:
        df = yf.download(tickers=i)
        df['Returns'] = df.apply(fdayGain, axis=1)
        df['Signal'] = df.apply(signal, axis=1)

        df['High Shifted'] = df['High'].shift(1)
        df['Low Shifted'] = df['Low'].shift(1)
        df['Close Shifted'] = df['Close'].shift(1)
        df['Upper BBand'], df['Middle BBand'], df['Lower BBand'] = ta.BBANDS(df['Close Shifted'], timeperiod=20)
        df.to_csv(path_or_buf = "./datasets/csv_{name}.csv".format(name = i))

if __name__ == '__main__':
    main()
