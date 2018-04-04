from __future__ import print_function
from keras.layers import Dense, Flatten, Reshape, LSTM, Activation, Dropout, Embedding, GRU, SimpleRNN
from keras.models import Sequential

import numpy as np
import unicodedata
import pickle
import random
import nltk
import sys
import os
import re

def basic_architecture(filtered_vocab, batch_size, sequence_length, parameter_updates = {}, is_stateful = False):
    parameter_dict = {}
    parameter_dict['hidden_units'] = 128
    parameter_dict['layers'] = 1
    parameter_dict.update(parameter_updates)
    model = Sequential()
    for i in range(parameter_dict['layers']):
        model.add(LSTM(parameter_dict['hidden_units'], stateful = is_stateful, return_sequences = True, implementation = 2, batch_input_shape = (batch_size, sequence_length, len(filtered_vocab)), input_shape=(batch_size, sequence_length, len(filtered_vocab))))
    model.add(Dense(len(filtered_vocab)))
    model.add(Activation('softmax'))
    return model

def twolayer_architecture(filtered_vocab, batch_size, sequence_length, parameter_updates = {}, is_stateful = False):
    parameter_dict = {}
    parameter_dict['hidden_units'] = 128
    parameter_dict.update(parameter_updates)
    model = Sequential()
    model.add(LSTM(parameter_dict['hidden_units'], stateful = is_stateful, return_sequences = True, implementation = 2, batch_input_shape = (batch_size, sequence_length, len(filtered_vocab)), input_shape=(batch_size, sequence_length, len(filtered_vocab))))
    model.add(Dense(len(filtered_vocab)))
    model.add(Activation('softmax'))
    return model
