import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import data


def simulation(ticker):
    """simulate day trading and compare to a buy and hold strategy"""

    model = svm.SVC(kernel="poly")
    df = data.preprocess(ticker)
    X_train, X_test, y_train, y_test = data.split(df)

    model.fit(X_train, y_train)

    # X = np.concatenate((X_train, X_test))
    X = np.array(df.drop(["Signal", "Returns"], axis=1))

    put_or_call = list(model.predict(df[df.columns[:-2]]))
    put_or_call = model.predict(X)

    print(list(put_or_call))
    quit()

    """
    something needs to be fixed here

    because of the shuffling effect of splitting the training and test set,
    how do you determine which predicted values should be associated with
    which prices/returns?
    
    """

    gains = [val * action for val, action in zip(diff, put_or_call)]
    net = sum(gains)

    return gains


def plot_simulation(ticker):

    figure: Figure = plt.figure()

    df = data.preprocess(ticker)
    value = list(df["Close Shifted"])

    """
    plot the net gain of the svm simulation
    """
    # gains = simulation(ticker)
    # plt.plot([i for i in range(len(value))], gains, label='Day Traded')

    plt.plot([i for i in range(len(value))], value, label="Buy & Hold")
    plt.plot([2500 for i in range((140))], [i for i in range((140))], label="Day Traded")

    plt.ylabel(ylabel="Value")
    plt.xlabel(xlabel="Time (Days)")
    plt.legend(loc="upper left")
    plt.title(f"Success of SVM on {ticker} Simulation (incomplete)")

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
