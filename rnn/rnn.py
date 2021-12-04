import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Embedding,
    InputLayer,
    LeakyReLU,
    SimpleRNN,
)
from tensorflow_addons.optimizers import CyclicalLearningRate
from tensorflow.keras.optimizers import SGD, Adam

import preprocess
from preprocess import rnn_data_pipeline, snp


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


def future(ndays: int):
    "predict a stock value n days into the future"
    pass


def graph(hist):
    "show results graphically"

    print(hist.keys())

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("RNN analysis on stock values", fontsize=16)

    ## accuracy
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(hist["accuracy"], label="Train acc.")
    ax.plot(hist["val_accuracy"], label="Validation acc.")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend(loc="lower right")
    ax.set_title("Accuracy of RNN")

    ## loss
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(hist["loss"], label="Train loss")
    ax.plot(hist["val_loss"], label="Validation loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(loc="lower left")
    ax.set_title("Loss of RNN")

    ## stock value
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Stock value")

    ## prediction
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("RNN prediction")

    # plt.show()
    fig.tight_layout()
    fig.savefig("results-rnn.png")
    fig.clf()


def main():
    "main analysis will occur here"

    # stocks = preprocess.snp()
    # X, y, SHAPE = preprocess.rnn_data_pipeline(stocks=stocks, timesteps=50)

    X, y = preprocess.load_npz()
    SHAPE = X[0].shape

    # X = X[0:2]
    # y = y[0:2]

    # model = build_model(kind="RNN", nunits=64, nlayers=1, bidirectional=True)

    ## building model
    tf.random.set_seed(1)

    model = Sequential(
        [
            ## input layer
            InputLayer(input_shape=SHAPE),
            ## recurrent layers
            LSTM(128, input_shape=SHAPE, return_sequences=True),
            Bidirectional(LSTM(32, return_sequences=True)),
            ## dense layers
            Dense(64, activation="selu"),
            Dense(16, activation="elu"),
            ## output layer
            Dense(1, activation="softmax"),
        ]
    )

    model.summary()

    # clr = CyclicalLearningRate(
    #     initial_learning_rate=0.01,
    #     maximal_learning_rate=0.1,
    #     scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    #     step_size=2 * steps_per_epoch,
    # )

    sgd = SGD(
        name="momentum-sgd",
        learning_rate=0.01, # could be clr
        momentum=0.4,
        nesterov=True,
        clipvalue=1.0,  # clipnorm...
    )

    adam = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"]
    )

    print(f"x shape: {X.shape}")
    print(f"y shape: {y.shape}")

    earlystopping = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=50, restore_best_weights=True
    )

    hist = model.fit(
        x=X,
        y=y,
        epochs=50,
        verbose=2,
        batch_size=SHAPE[0],
        validation_split=0.2,
        callbacks=[earlystopping],
    )
    hist = hist.history
    # no more than 86 epochs needed?

    graph(hist)


if __name__ == "__main__":
    main()


"""
NOTES:
is binary classification suitable?
    linear regression
----------

TODO:
- node dropout layer
- shuffle the data
- train on stock returns not stock signal?
- add more technical indicators


l2 l1 regularization
golden test : 1 or 2 samples

gradient tape


"""
