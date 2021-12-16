import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib as ta
import yfinance as yf
from sklearn import metrics, preprocessing, svm
from sklearn.model_selection import train_test_split
from talib import MA_Type
from tqdm import tqdm

import data
import JMaffet.RIFSVM as RIFSVM

'''
the program doesnt run:
----------
RIFSVM' has no attribute 'simulation'
'''


def percentgain(gain, start):
    """
    not called in project
    """
    percentgains = []
    for i in range(5):
        percentgains.append(BHReturn(gain[i], start[i]))
    return percentgains


def model(df:DataFrame) -> int:

    ## build model
    model = svm.SVC(kernel="poly")

    ## normalize df
    normalized_df = (df - df.mean()) / df.std()
    normalized_df["Signal"] = df["Signal"]
    normalized_df = normalized_df.drop(
        ["High", "Low", "Close", "Adj Close", "Returns"], axis=1, inplace=False
    )

    ## ??? could be this be called from RIFSVM.model(df)
    modern_predict = normalized_df.iloc[4864:]
    normalized_df = normalized_df.iloc[:-365]
    print(normalized_df)
    print(modern_predict)

    ModernX_test = np.array(modern_predict[modern_predict.columns[:-2]])
    # Moderny_test = np.array(normalized_df[normalized_df.columns[-1]])

    ## split df
    train, test = train_test_split(normalized_df, test_size=0.2)

    X_train = np.array(train[train.columns[:-2]])
    y_train = np.array(train[train.columns[-1]])

    X_test = np.array(test[test.columns[:-2]])
    y_test = np.array(test[test.columns[:-1]])

    ## predictions
    model.fit(X_train, y_train)

    y_pred = list(model.predict((ModernX_test)))
    y_pred = [1 if i == 1 else -1 for i in y_pred]
    gain_value = list(df["Gain Value"])

    gains = [val * action for val, action in zip(gain_value, y_pred)]
    net = sum(gains)

    cumulative_gains = []
    total = 0
    for item in gains:
        total += item
        cumulative_gains += [total]

    return sum(cumulative_gains)


def main():

    print(len(RIFSVM.simulation(data.preprocess("AAPL"))))

    ## returns the sum of gains according to a trading strategy
    print(model(data.preprocess("AAPL")))


if __name__ == "__main__":
    main()
