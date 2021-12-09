import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import data


def simulation(df):
    """simulate day trading and compare to a buy and hold strategy"""

    model = svm.SVC(kernel="poly")

    normalized_df=(df-df.mean())/df.std()

    normalized_df["Signal"] = df["Signal"]
    normalized_df = normalized_df.drop(["High", "Low", "Close", "Adj Close", "Returns"], axis=1, inplace=False)

    train, test = train_test_split(normalized_df, test_size=0.2)

    X_train = np.array(train[train.columns[:-2]])
    y_train = np.array(train[train.columns[-1]])

    X_test = np.array(test[test.columns[:-2]])
    y_test = np.array(test[test.columns[-1]])

    model.fit(X_train, y_train)

    y_pred = list(model.predict(np.array(normalized_df[normalized_df.columns[:-2]])))
    y_pred = [1 if i==1 else -1 for i in y_pred]
    gain_value = list(df['Gain Value'])


    gains = [val * action for val, action in zip(gain_value, y_pred)]
    net = sum(gains)

    cumulative_gains = []
    total = 0
    for item in gains:
        total += item
        cumulative_gains += [total]

    return cumulative_gains


def plot_simulation(ticker, n=100):

    figure: Figure = plt.figure()

    df = data.preprocess(ticker)
    value = list(df["Close Shifted"])


    plt.plot([i for i in range(len(value))], value, label="Buy & Hold")


    gains = simulation(df)
    plt.plot([i for i in range(len(gains))], gains, 'r', alpha=0.3, label='Day Traded')

    for i in tqdm(range(n)):
        gains = simulation(df)
        plt.plot([i for i in range(len(gains))], gains, 'r', alpha=0.3)


    plt.ylabel(ylabel="Value")
    plt.xlabel(xlabel="Time (Days)")
    plt.legend(loc="upper left")
    plt.title(f"Success of SVM on {ticker} Simulation")

    # plt.show()
    figure.savefig("simulation-svm.png")
    figure.clf()


def plot_accuracy(ticker, model, n=10):
    """distribution of scores over {n} iterations"""

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
    figure.savefig("accuracy-svm.png")
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
    plt.savefig("confusion-matrix-svm.png")
    plt.clf()


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
