"""

I dont think that there is anything in this file which isnt redundant in
data.py or svm.py

The structure is a little disorganized and hard to read.

It would be best to delete before project submission but I havent done
it yet incase there was any code anyone needed inside...

"""



import numpy as np
import yfinance as yf

import talib as ta
from talib import MA_Type

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.dummy import DummyClassifier


def fadd(x):
    dollar_gain = x["Close"] - x["Open"]
    return dollar_gain


def signal(x):
    if x["Returns"] >= 0:
        return 1
    else:
        return 0


df = yf.download(tickers="AAPL")

df["Returns"] = df.apply(fadd, axis=1)
df["Signal"] = df.apply(signal, axis=1)

df["High Shifted"] = df["High"].shift(1)
df["Low Shifted"] = df["Low"].shift(1)
df["Close Shifted"] = df["Close"].shift(1)
df["Upper BBand"], df["Middle BBand"], df["Lower BBand"] = ta.BBANDS(
    df["Close Shifted"], timeperiod=20
)
df["Upper BBand"], df["Middle BBand"], df["Lower BBand"] = ta.BBANDS(
    df["Close Shifted"], timeperiod=20
)



max_abs_scaler = preprocessing.MaxAbsScaler()
df.dropna(inplace=True)

print(df)

X = np.array(df.drop(["Signal", "Returns"], 1))
X = max_abs_scaler.fit_transform(X)
Y = np.array(df["Signal"])

X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y, test_size=0.15)
X2_train, X1_dev, y2_train, y1_dev = train_test_split(
    X1_train, y1_train, test_size=0.15
)

def predict():
    model = svm.SVC(kernel="poly")  # Best performing

    # ---- Evaluations on dev training and dev test
    model.fit(X2_train, y2_train)
    y1_pred = model.predict(X1_dev)

    print("Accuracy: ", metrics.accuracy_score(y1_dev, y1_pred))
    print("Precision: ", metrics.precision_score(y1_dev, y1_pred, average="macro"))

    # y2_pred = knn.predict(X2_train, y2_train, X1_dev, 7)
    # print(metrics.accuracy_score(y1_dev, y2_pred))

    # ---- Actual evaluations on whole training and whole test
    model.fit(X1_train, y1_train)
    y3_pred = model.predict(X1_test)

    print("Accuracy: ", metrics.accuracy_score(y1_test, y3_pred))
    print("Precision: ", metrics.precision_score(y1_test, y3_pred, average="macro"))

    # y4_pred = knn.predict(X1_train, y1_train, X1_test , 7)
    # print(metrics.accuracy_score(y1_test, y4_pred))

    # # ---- Dummy Classifier
    # # Stratified performs worse
    # dummy_clf = DummyClassifier(strategy="most_frequent")
    # dummy_clf.fit(X1_train, y1_train)
    # d_predict = dummy_clf.predict(X1_test)
    # print(dummy_clf.score(X1_test, y1_test))

def main():
    pass

if __name__ == '__main__':
    main()
