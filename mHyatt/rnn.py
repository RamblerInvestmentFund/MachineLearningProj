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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import preprocess
from preprocess import rnn_data_pipeline, snp

"""
TODO
predict all features

fontend = df...
for i in n_steps:

    tomorrow = []
    for f in feature:
        model.fit(df[!f], df[f])
        ft = model.predict(df[!f])
        tomorrow += [ft]

    frontend += [tomorrow]
    frontend.remove(0)
"""


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

def plot_confusion_matrix(y_test, y_pred):

    ## init graph
    fig = plt.figure()

    cm = confusion_matrix(y_test, y_pred, labels=[1,0])
    cm_display = ConfusionMatrixDisplay(y_test, y_pred).plot()

    plt.title(f"Confusion Matrix of RNN on S&P 500")
    plt.savefig(f"confusion-matrix-rnn.png")
    plt.clf()

def main():
    "main analysis will occur here"

    # preprocess.save_npz()
    print('download complete stock data? eta 5 min...')
    if input('(y/n) ').lower() == 'y':
        try:
            preprocess.save_npz()
            X, y = preprocess.load_npz()
            SHAPE = X[0].shape
        except:
            X, y, SHAPE = preprocess.rnn_data_pipeline(["SPY"])
    else:
        X, y, SHAPE = preprocess.rnn_data_pipeline(["SPY"])

    ## building model
    tf.random.set_seed(1)

    model = Sequential(
        [
            ## input layer
            InputLayer(input_shape=SHAPE),
            ## recurrent layers
            LSTM(32, input_shape=SHAPE, return_sequences=True),
            Bidirectional(LSTM(32, return_sequences=True)),
            ## dense layers
            Dense(64, activation="selu"),
            Dense(16, activation="selu"),
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
        learning_rate=0.01,  # could be clr
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

    ## validation data
    val_split = 0.2
    val_size = int(val_split*len(X))

    val = (X[-val_size:],y[-val_size:])

    X = X[:-val_size]
    y = y[:-val_size]


    print(f"x shape: {X.shape}")
    print(f"y shape: {y.shape}")

    earlystopping = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    )

    hist = model.fit(
        x=X,
        y=y,
        epochs=50,
        verbose=2,
        batch_size=SHAPE[0],
        validation_data=val,
        callbacks=[earlystopping],
    )
    hist = hist.history

    graph(hist)

    # y_pred = model.predict(val[0])
    # confusion = [list(np.reshape(val[1], (1,-1)).values.argmax(axis=1)),list(np.reshape(y_pred, (1,-1)).values.argmax(axis=1))]
    # print(type(confusion[0]), type(confusion[1]))
    # plot_confusion_matrix( confusion[0], confusion[1] )


if __name__ == "__main__":
    main()


"""
NOTES:
is binary classification suitable?
    linear regression
----------

TODO:
- node dropout layer
- shuffle the data (batches)
- train on stock returns not stock signal?
- add more technical indicators


gradient tape


"""
