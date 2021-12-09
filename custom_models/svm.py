from sklearn import metrics, preprocessing, svm
from sklearn.model_selection import train_test_split

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

import data

def simulation(ticker):
    '''simulate day trading and compare to a buy and hold strategy'''

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

    net = int(sum([val * action for val, action in zip(diff, put_or_call)]))

    if net in dollars.keys():
        dollars[net] += 1
    else:
        dollars[net] = 1



def plot_simulation(ticker):

    df = data.preprocess(ticker)
    value = list(df["Close Shifted"])


    figure: Figure = plt.figure()

    plt.plot([i for i in range(len(value))], value, label='Buy & Hold')

    plt.ylabel(ylabel="Value")
    plt.xlabel(xlabel="Time (Days)")
    # plt.xticks()

    plt.legend(loc="upper left")


    plt.title(f"Success of SVM on {ticker} Simulation")

    # plt.show()
    figure.savefig("simulation-svm.png")
    figure.clf()

def plot_f1(ticker):
    scores = {}
    df = data.preprocess(ticker)

    X_train, X_test, y_train, y_test = data.split(df)
    model = svm.SVC(kernel="poly")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(list(y_pred))
    quit()
    score = metrics.accuracy_score(y_test, y_pred)



def plot_accuracy(ticker, model, n=10000):
    '''distribution of scores over {n} iterations'''

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


def assess(model, X_test, y_test):
    """determines effectiveness of the model"""

    y_pred = model.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred, average="macro"))
    print("Recall: ", metrics.recall_score(y_test, y_pred, average="macro"))

def main():

    stocks = ["AAPL", "MSFT", "SPY", "QQQ", "DIA", "TLT", "GLD", "CVX", "KO", "PEP", "PG", "JNJ", "GSK"]
    model = svm.SVC(kernel="poly")

    for ticker in stocks:

        df = data.preprocess(ticker)
        X_train, X_test, y_train, y_test = data.split(df)

        model.fit(X_train, y_train)

        plot_accuracy(ticker, model)

        y_pred = assess(model, X_test, y_test)
        y_pred = assess(model, X_train, y_train)


        quit()

if __name__ == "__main__":
    main()
