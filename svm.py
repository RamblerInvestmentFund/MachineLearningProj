import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import data

'''
maybe it would be better for the model
to be trained on the first x% of data
rather than a random x%

since it is a sequential model there is no reason to predict
returns that are before other returns we (users) already know

ie: if i know the value of a stock in 2020 i dont need to predict
the ones from 2019 etc cuz its not useful to trading
'''


def simulation(df, model):
    """simulate day trading and compare to a buy and hold strategy"""

    # maybe min-max normalized is better ... ask anthony
    normalized_df=(df-df.mean())/df.std()

    normalized_df["Signal"] = df["Signal"]
    normalized_df = normalized_df.drop(["High", "Low", "Close", "Adj Close", "Returns"], axis=1, inplace=False)

    train, test = train_test_split(normalized_df, test_size=0.2)

    X_train = np.array(train[train.columns[:-2]])
    y_train = np.array(train[train.columns[-1]])

    X_test = np.array(test[test.columns[:-2]])
    y_test = np.array(test[test.columns[-1]])

    model.fit(X_train, y_train)


    # predicting net value
    gain_value = list(df['Gain Value'])
    y_pred = list(model.predict(np.array(normalized_df[normalized_df.columns[:-2]])))

    y_pred = [1 if i==1 else -1 for i in y_pred]

    # interpretting class labels ... might need a double check
    gain1 = [val * action for val, action in zip(gain_value, y_pred)]
    gain2 = [val * -action for val, action in zip(gain_value, y_pred)]

    gains = gain1 if sum(gain1) > sum(gain2) else gain2

    ## perfect guesser
    # gains = [abs(val) for val in gain_value]

    net = sum(gains)

    cumulative_gains = []
    total = 0
    for item in gains:
        total += item
        cumulative_gains += [total]

    return cumulative_gains


def plot_simulation(ticker, n=10):

    figure: Figure = plt.figure()

    model = svm.SVC(kernel="poly")

    df = data.preprocess(ticker)
    value = list(df["Close Shifted"])
    value = [v - value[0] for v in value]

    gains = simulation(df, model)
    plt.plot([i for i in range(len(gains))], gains, 'r', alpha=0.3, label='Day Traded')

    avg_gains = []
    for i in tqdm(range(n)):
        gains = simulation(df, model)
        plt.plot([i for i in range(len(gains))], gains, 'r', alpha=0.3)
        avg_gains += [gains[-1]]
    avg_gains = int(sum(avg_gains) / len(avg_gains))

    print(f'Average gains on {ticker}:      {avg_gains}')
    print(f'Buy and Hold Gains on {ticker}: {int(value[-1])}')
    print()


    plt.plot([i for i in range(len(value))], value, label="Buy & Hold")

    '''idea:
    plot average of day trades as a darker red line
    '''

    assess(df, model)

    plt.ylabel(ylabel="Value")
    plt.xlabel(xlabel="Time (Days)")
    plt.legend(loc="upper left")
    plt.title(f"Success of SVM on {ticker} Simulation")

    # plt.show()
    figure.savefig(f"{ticker}-simulation-svm.png")
    figure.clf()


def plot_accuracy(ticker, n=10):
    """distribution of scores over {n} iterations"""

    model = svm.SVC(kernel="poly")

    scores = {}
    df = data.preprocess(ticker)

    for i in tqdm(range(n)):

        X_train, X_test, y_train, y_test = data.split(df)
        model = svm.SVC(kernel="poly")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)

        score = int(score * 100) / 100

        if score in scores.keys():
            scores[score] += 1
        else:
            scores[score] = 1

    figure: Figure = plt.figure()

    keys = [key for key in scores.keys()]
    plt.ylabel(ylabel="Frequency")
    plt.xlabel(xlabel="Accuracy")
    plt.xticks(ticks=keys, labels=keys)
    plt.title(f"Accuracy of SVM on {ticker}")
    plt.bar(scores.keys(), scores.values(), width=0.009)

    # plt.show()
    figure.savefig(f"{ticker}-accuracy-svm.png")
    figure.clf()


def plot_confusion_matrix(ticker):

    model = svm.SVC(kernel="poly")

    ## init data
    df = data.preprocess(ticker)
    X_train, X_test, y_train, y_test = data.split(df)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    ## init graph
    fig = plt.figure()

    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.plot(cmap=plt.cm.Blues)

    plt.title(f"Confusion Matrix of SVM on {ticker}")
    plt.savefig(f"{ticker}-confusion-matrix-svm.png")
    plt.clf()

def assess(df, model):
    'assesses effectiveness of a stock ticker'

    X_train, X_test, y_train, y_test = data.split(df)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred, average="macro"))
    print("Recall: ", metrics.recall_score(y_test, y_pred, average="macro"))


def main():

    stocks = [
        "AAPL",
        "MSFT",
        "SPY",
        "QQQ",
    ]
    # [ "DIA", "TLT", "GLD", "CVX", "KO", "PEP", "PG", "JNJ", "GSK"]

    model = svm.SVC(kernel="poly")

    for ticker in stocks:

        df = data.preprocess(ticker)
        X_train, X_test, y_train, y_test = data.split(df)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        print("Precision: ", metrics.precision_score(y_test, y_pred, average="macro"))
        print("Recall: ", metrics.recall_score(y_test, y_pred, average="macro"))


if __name__ == "__main__":
    main()
