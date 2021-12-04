import yfinance as yf
import numpy as np

import talib as ta
from talib import MA_Type

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

'create datasets from yahoo finance'
'preprocesses data for analysis '

def signal(x):
    return 1 if x['Returns'] >= 0 else 0

def main():
    max_abs_scaler = preprocessing.MaxAbsScaler()

    stocks = ["AAPL", "MSFT", "SPY", "QQQ", "DIA", "TLT", "GLD", "CVX", "KO", "PEP", "PG", "JNJ", "GSK"]
    
    for i in stocks:
        df = yf.download(tickers=i, start='2001-01-01')

        df['High Shifted'] = df['High'].shift(1)
        df['Low Shifted'] = df['Low'].shift(1)
        df['Close Shifted'] = df['Close'].shift(1)

        df['Upper BBand'], df['Middle BBand'], df['Lower BBand'] = ta.BBANDS(df['Close Shifted'], timeperiod=20)
        df['RSI'] = ta.RSI(np.array(df['Close Shifted']), timeperiod=14)
        df['Macd'], df['Macd Signal'],df['Macd Hist'] = ta.MACD(df['Close Shifted'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['Momentum'] = ta.MOM(df['Close Shifted'],timeperiod=12)
        
        df['Returns'] = np.log(df['Open']/df['Open'].shift(1)) 
        df['Signal'] = df.apply(signal, axis=1)
        df = df[40:]

        df.to_csv(path_or_buf = "./datasets/csv_{name}.csv".format(name = i))
    
        X = np.array(df.drop(["Signal", "Returns"], 1))
        X = max_abs_scaler.fit_transform(X)
        Y = np.array(df["Signal"])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
        model = svm.SVC(kernel="poly")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        print("Precision: ", metrics.precision_score(y_test, y_pred, average="macro"))
        print("Recall: ", metrics.recall_score(y_test, y_pred, average="macro"))

if __name__ == '__main__':
    main()
