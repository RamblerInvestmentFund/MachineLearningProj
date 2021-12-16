import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import mHyatt.data as data

'''
how were the graphs produced?
'''

def model(df):
    '''
    returns svm predictions on a stock ticker
    y_pred
    '''

    ## build model
    model = svm.SVC(kernel="poly")
    normalized_df = (df - df.mean()) / df.std()

    ## normalize df
    normalized_df["Signal"] = df["Signal"]
    normalized_df = normalized_df.drop(
        ["High", "Low", "Close", "Adj Close", "Returns"], axis=1, inplace=False
    )

    ## ???
    modern_predict = normalized_df.iloc[4864:]
    normalized_df = normalized_df.iloc[:-365]
    modern_predict_array = np.array(modern_predict[modern_predict.columns[:-2]])

    ## split df
    train, test = train_test_split(normalized_df, test_size=0.2)

    X_train = np.array(train[train.columns[:-2]])
    y_train = np.array(train[train.columns[-1]])

    X_test = np.array(test[test.columns[:-2]])
    y_test = np.array(test[test.columns[-1]])

    ## predictions
    model.fit(X_train, y_train)

    y_pred = list(model.predict(modern_predict_array))
    y_pred = [1 if i == 1 else -1 for i in y_pred]
    return y_pred


def strategy(df, prediction):
    '''
    implements a trading strategy on a stock ticker
    (specify what the strategy is?)
    '''

    ## init
    money = 50
    day_buy = 0
    flag = 1 # what does the flag do

    open = list(df["Open"])
    close = list(df["Close"])

    for i in range(len(prediction)):
        if flag == 1:
            if prediction[i] == -1:
                money = money * (close[i]) / open[day_buy]
                flag = 0
        elif flag == 0:
            if prediction[i] == 1:
                flag = 1
                day_buy = i

    percent_return = (money / 50 * 100) - 100
    return percent_return


def main():
    '''
    main method:
    applies a trading strategy on 100 SVMs
    prints the average
    '''

    df = data.preprocess("JPM")

    ## what is year_df? ... everything but the first 4864 years of df?
    year_df = df.iloc[4864:]

    values = []

    for i in tqdm(range(100)):

        ## train and predict an svm
        vals = model(df)

        ## apply trading strategy
        values.append(strategy(year_df, vals))

    print([int(v) for v in values]) # printed as int for clarity
    print(sum(values) / len(values))


if __name__ == "__main__":
    main()
