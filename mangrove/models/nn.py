# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from app.application_context import context
from numpy import ndarray
from util.string_util import is_empty

__author__ = "ujihirokazuya"
__date__ = "2017/05/14"


class NeuralNetwork(object):

    def __init__(self, x: ndarray):
        self._config = context.config
        train_data_shape = x.shape
        self._sequence_length = train_data_shape[1]
        self._block_size = train_data_shape[2]

    def create_lstm_nn(self, weights_file_path) -> Sequential:
        model = Sequential()
        model = self._add_lstm(model)
        model.add(TimeDistributed(Dense(self._block_size),
                                  input_shape=(self._sequence_length, self._config.hidden_dimension_size)))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        # model.add(TimeDistributed(Dense(num_frequency_dimensions, activation="sigmoid"), input_shape=(40,
        # num_hidden_dimensions)))
        # model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        # model.add(TimeDistributed(Dense(num_frequency_dimensions, activation="softmax"), input_shape=(40,
        # num_hidden_dimensions)))
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        if not is_empty(weights_file_path):
            model.load_weights(filepath=weights_file_path)
        return model

    def _add_lstm(self, model: Sequential) -> Sequential:
        hidden_dimension_size = self._config.hidden_dimension_size
        model.add(LSTM(hidden_dimension_size,
                       input_shape=(self._sequence_length, self._block_size),
                       return_sequences=True))
        recurrent_unit_size = self._config.recurrent_unit_size
        if recurrent_unit_size <= 1:
            return model
        for cur_unit in range(self._config.recurrent_unit_size - 1):
            model.add(LSTM(hidden_dimension_size,
                           input_shape=(self._sequence_length, hidden_dimension_size),
                           return_sequences=True))
        return model
