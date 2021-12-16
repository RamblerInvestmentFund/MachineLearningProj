import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import data
import statistics

def model(df, timeframe):
    model = svm.SVC(kernel="poly")
    normalized_df = (df - df.mean()) / df.std()

    normalized_df["Signal"] = df["Signal"]
    normalized_df = normalized_df.drop(["High", "Low", "Close", "Adj Close", "Returns"], axis=1, inplace=False)

    modern_predict = normalized_df.iloc[5229-timeframe:]
    normalized_df = normalized_df.iloc[:-timeframe]
    modern_predict_array = np.array(modern_predict[modern_predict.columns[:-2]])

    train, test = train_test_split(normalized_df, test_size=0.2)

    X_train = np.array(train[train.columns[:-2]])
    y_train = np.array(train[train.columns[-1]])

    X_test = np.array(test[test.columns[:-2]])
    y_test = np.array(test[test.columns[-1]])

    model.fit(X_train, y_train)

    y_pred = list(model.predict(modern_predict_array))
    y_pred = [1 if i == 1 else -1 for i in y_pred]
    return y_pred

def stratagy(df, prediction):
    money = 50
    day_buy = 0
    flag = 1
    open = list(df["Open"])
    close = list(df["Close"])
    for i in range(len(prediction)):
        if flag == 1:
            if prediction[i] == -1:
                money = money * (close[i])/open[day_buy]
                flag = 0
        elif flag == 0:
            if prediction[i] == 1:
                flag = 1
                day_buy = i


    percent_return = (money/50*100) - 100
    return percent_return


def run_model(ticker,numsimulations,timeframe):
    df = data.preprocess(ticker)
    timed_df = df.iloc[5229-timeframe:]
    values = []
    for i in range(numsimulations):
        vals = model(df,timeframe)
        values.append(stratagy(timed_df, vals))
    return values







