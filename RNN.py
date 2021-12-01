import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense


def generic_model():
    """
    borrowed from the textbook
    bidirectional LSTM
    """

    bi_lstm = tf.keras.Sequential(
        [
            Embedding(input_dim=1000, output_dim=32),
            Bidirectional(LSTM(64, name="lstm-layer")),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    bi_lstm.sumary()

    return bi_lstm

    # bi_lstm.compile(**kwargs)
    # hist = bi_lstm.fit(*data, **kwargs)


def build_model(embedding_dim, vocab_size, kind="RNN", nunits=64, nlayers=1, bidirectional=True):
    """
    borrowed from textbook
    used to test many different types of RNNs
    """

    tf.random.set_seed(1)

    # build the model
    model = tf.keras.Sequential()

    model.add(
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embed-layer")
    )

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


def main():
    'main analysis will occur here'
    pass


if __name__ == "__main__":
    main()
