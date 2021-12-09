from sklearn import metrics, preprocessing, svm
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import matplotlib.pyplot as plt

import data

def simulation():
    '''simulate day trading and compare to a buy and hold strategy'''

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
