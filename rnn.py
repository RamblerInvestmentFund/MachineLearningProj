import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer

from sklearn.preprocessing import StandardScaler


def build_model(input=(54,), kind="RNN", nunits=64, nlayers=1, bidirectional=True):
    """
    borrowed from textbook
    used to test many different types of RNNs
    """

    tf.random.set_seed(1)

    # build the model
    model = tf.keras.Sequential()

    kind = kind.upper()  # prevents user error

    # add recurrent layers
    for i in range(nlayers):
        sequences = i < nlayers - 1

        if kind == "RNN":
            layer = SimpleRNN(
                units=nunits, return_sequences=sequences, name=f"rnn-layer{i}"
            )
        if kind == "LSTM":
            layer = LSTM(
                units=nunits, return_sequences=sequences, name=f"lstm-layer{i}"
            )
        if kind == "GRU":
            layer = GRU(units=nunits, return_sequences=sequences, name=f"gru-layer{i}")

        if bidirectional:
            layer = Bidirectional(layer, name=f"bidir-{layer.name}")

        model.add(layer)

    # add dense layers
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()
    return model


def _clean(path: str):
    "preprocesses stock data"

    df = pd.read_csv(path)
    df = df.drop(columns=["High", "Low", "Adj Close", "Volume"])
    # maybe add volume back in later ...
    # or for doing length of gradient rather than direction

    for i in range(53):
        if i != 0:
            df[f"Returns-{i}"] = df["Returns"].shift(i)

    df["52-week-high"] = df[[c for c in df.columns if "Returns-" in c]].max(axis=1)
    # use days since 52 week high?

    df = df.drop(columns=["Date", "Close", "Returns"])

    ## droped most of excess data
    col = [f'Returns-{i}' for i in range(15,52)]
    df = df.drop(columns=[*col])

    df.dropna(inplace=True)

    X = df
    y = X.pop("Signal")

    stdsc = StandardScaler()
    X = stdsc.fit_transform(X)

    SIZE = len(y.T)

    ## for visualizing the dataset

    # pd.set_option('display.max_rows', None)
    # print(std_df)
    # print(df.columns)

    return X, y, SIZE


def preprocess(path: str):

    X, y, SIZE = _clean(path)

    X = np.array(X)
    y = np.array(y.values)

    'convert to tensor?'

    ## splitting the dataset
    # train_size = int(0.7 * SIZE)
    # val_size = int(0.15 * SIZE)
    # test_size = int(0.15 * SIZE)
    #
    # train = dataset.take(train_size)
    # test = dataset.skip(train_size)
    #
    # val = test.skip(val_size)
    # test = test.take(test_size)
    #
    # return train, val, test
    return X, y
    return dataset


def future(ndays: int):
    "predict a stock value n days into the future"
    pass


def reshape():
    "compartmentalize reshaping of data"
    pass

def graph(hist):
    'show results graphically'

    print(hist.keys())

    fig = plt.figure(figsize=(12,8))
    fig.suptitle('RNN analysis on stock values', fontsize=16)


    ## accuracy
    ax = fig.add_subplot(2,2,1)
    ax.plot(hist['accuracy'], label='Train acc.')
    ax.plot(hist['val_accuracy'], label='Validation acc.')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(loc='lower right')
    ax.set_title('Accuracy of RNN')


    ## loss
    ax = fig.add_subplot(2,2,2)
    ax.plot(hist['loss'], label='Train loss')
    ax.plot(hist['val_loss'], label='Validation loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='lower left')
    ax.set_title('Loss of RNN')


    ## stock value
    ax = fig.add_subplot(2,2,3)
    ax.set_title('Stock value')


    ## prediction
    ax = fig.add_subplot(2,2,4)
    ax.set_title('RNN prediction')




    # plt.show()
    fig.tight_layout()
    fig.savefig('results-rnn.png')
    fig.clf()




def main():
    "main analysis will occur here"

    print(tf.version)

    X, y = preprocess("datasets/csv_AAPL.csv")

    # model = build_model(kind="RNN", nunits=64, nlayers=1, bidirectional=True)

    ## preparing model
    batch = 2566 # 2566/4 steps 5132/8 steps
    timesteps = int(10264 / batch)
    feature = 17
    SHAPE = (timesteps, feature)
    print(SHAPE)

    print(X.shape)

    X = np.reshape(X, newshape=(batch, timesteps, feature))
    y = np.reshape(y, newshape=(batch, timesteps, 1))

    print()
    print(y.shape)
    print(X.shape)

    ## building model
    tf.random.set_seed(1)
    model = Sequential()
    model.add(LSTM(64, input_shape=SHAPE, return_sequences=True))
    model.add(Bidirectional(LSTM(16, return_sequences=True)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(
        optimizer="SGD", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"]
    )

    print(f"x shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print()
    hist = model.fit(x=X, y=y, epochs=100, verbose=2, validation_split=0.2)
    hist = hist.history

    graph(hist)


if __name__ == "__main__":
    main()


'''

Notes
what is the information problem?
trend analysis
    slope of stock

long term
    weekly

is binary classification suitable?

maybe 52 days is too much
using sector as a feature
use linear regression to derive a direction

'''
