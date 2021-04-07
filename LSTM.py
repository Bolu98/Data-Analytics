#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import array
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

"""
Created on Mon Mar 22 10:34:44 2021

@author: akosreitz
"""


def predictionFunction(days=1):
    """Shows the anticipated next closing price of IBM common stock on 25/03/2021"""
    load = pd.read_csv(r"IBM_csv")
    temp = (load["Close"].tolist())[::-1]
    tf.config.threading.set_intra_op_parallelism_threads(2)

    def split_sequence(sequence, n_steps):
        x, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            x.append(seq_x)
            y.append(seq_y)
        return array(x), array(y)

    x, y = split_sequence(temp, 30)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    model = Sequential()
    model.add(LSTM(30, activation="relu", input_shape=(30, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=100, batch_size=30, verbose=1)
    x_input = (((load.iloc[-30::])["Close"]).to_list())[::-1]
    x_input = array(x_input).reshape(1, 30, 1)
    yhat = model.predict(x_input, verbose=0)
    return yhat


print(predictionFunction(1))
