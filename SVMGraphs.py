import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import SVMRIF
import Data
import pandas as pd
import numpy as np
import talib as ta
from talib import MA_Type
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm


print(len(SVMRIF.simulation(Data.preprocess('AAPL'))))


def percentgain(gain,start):
    percentgains = []
    for i in range(5):
        percentgains.append(BHReturn(gain[i],start[i]))
    return percentgains


def model(df):
    model = svm.SVC(kernel="poly")

    normalized_df=(df-df.mean())/df.std()

    normalized_df["Signal"] = df["Signal"]
    normalized_df = normalized_df.drop(["High", "Low", "Close", "Adj Close", "Returns"], axis=1, inplace=False)

    modern_predict = normalized_df.iloc[4864:]
    normalized_df = normalized_df.iloc[:-365]
    print(normalized_df)
    print(modern_predict)

    ModernX_test = np.array(modern_predict[modern_predict.columns[:-2]])
    #Moderny_test = np.array(normalized_df[normalized_df.columns[-1]])

    train, test = train_test_split(normalized_df, test_size=0.2)

    X_train = np.array(train[train.columns[:-2]])
    y_train = np.array(train[train.columns[-1]])

    X_test = np.array(test[test.columns[:-2]])
    y_test = np.array(test[test.columns[:-1]])

    model.fit(X_train, y_train)

    y_pred = list(model.predict((ModernX_test)))
    y_pred = [1 if i==1 else -1 for i in y_pred]
    gain_value = list(df['Gain Value'])


    gains = [val * action for val, action in zip(gain_value, y_pred)]
    net = sum(gains)

    cumulative_gains = []
    total = 0
    for item in gains:
        total += item
        cumulative_gains += [total]

    return sum(cumulative_gains)

print(model(Data.preprocess('AAPL')))