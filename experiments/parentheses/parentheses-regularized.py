# 1 ---------------------------------------------------------------------------
import sys
sys.path.append('../../src')

# 2 ---------------------------------------------------------------------------

N_SENTENCES = 50000
TEST_RATIO = .1
WINDOW_SIZE = 16
N_EPOCHS = 5
CACHED = True
N_STATES = 32
SAMPLES_VERIFY = 8
BATCH_SIZE = 32

# 3 ---------------------------------------------------------------------------

import datetime
import os

FOLDER_OUT = 'Parenthesis-' + str(datetime.datetime.today())
if not os.path.exists(FOLDER_OUT):
    os.makedirs(FOLDER_OUT)
# 4 ---------------------------------------------------------------------------

import numpy as np
import keras

# 6 ---------------------------------------------------------------------------
import random
random.seed(55555)

# 7 ---------------------------------------------------------------------------

# Creates dataset
import ipdb; ipdb.set_trace()

import utils.grammar as gram
reload(gram)
grammar = [('S0', '0 S0 | ( S1 )'),
           ('S1', '1 S1 | ( S2 )'),
           ('S2', '2 S2 | ( S3 )'),
           ('S3', '3 S3 | ( S4 )'),
           ('S4', 'S4 S4 | 4')]
gram_obj = gram.GrammarUseCase(grammar, 'S0')

# 8 ---------------------------------------------------------------------------

import utils.preprocess as pre
reload(pre)

# Generates data
train_sequence = gram_obj.gen_sequence(N_SENTENCES)
test_sequence = gram_obj.gen_sequence(int(N_SENTENCES * TEST_RATIO))

raw_train_sequence, raw_test_sequence, char2int, int2char = pre.encode_split(train_sequence, test_sequence)

print 'Training smybols:', len(raw_train_sequence)
print 'Testing smybols:', len(raw_test_sequence)
print 'Size vocabulay:', len(char2int)

# 9 ---------------------------------------------------------------------------

import numpy as np
import utils.preprocess as pre
reload(pre)

# TODO: change this to regression labels.

import ipdb; ipdb.set_trace()


# Preprocesses training
# SPLITS
X_train_raw = raw_train_sequence[:-1]
y_train_raw = raw_train_sequence[1:]
# ENCODES
X_train = keras.utils.to_categorical(X_train_raw)
y_train = keras.utils.to_categorical(y_train_raw)
# SHUFFLES
train_indices = pre.shuffle_indices(len(X_train_raw), BATCH_SIZE)
y_train = y_train[train_indices]
# EXPANDS
X_train = X_train[train_indices,np.newaxis,:]

# Preprocesses testing
# SPLITS
X_test_raw = raw_test_sequence[:-1]
y_test_raw = raw_test_sequence[1:]
# ENCODES
X_test = keras.utils.to_categorical(X_test_raw)
y_test = keras.utils.to_categorical(y_test_raw)
# SHUFFLES
test_indices = pre.shuffle_indices(len(X_test_raw), BATCH_SIZE)
X_test = X_test[test_indices, np.newaxis,:]
y_test = y_test[test_indices]

print "Training data:"
print "X:", X_train.shape
print "y:", y_train.shape

print "Test data:"
print "X:", X_test.shape
print "y:", y_test.shape


# Recovers the original training data
X_test_sequence = test_sequence[:X_test.shape[0]]
y_test_sequence = test_sequence[1:X_test.shape[0]+1]
print 'Test sequences:', X_test_sequence[:5], y_test_sequence[:5]
print 'length:', len(X_test_sequence), len(y_test_sequence)

# 10 --------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,SimpleRNN 
from keras.layers import Lambda
from keras import regularizers

from controllers.mylstm_legacy import MYLSTM

in_dim = X_train.shape[1:]
out_dim = y_train.shape[1]

model = Sequential()
model.add(MYLSTM(N_STATES, stateful=True,
                         batch_size=BATCH_SIZE,
                           input_shape=in_dim,
                          activity_regularizer = regularizers.l1(0.01)))

# TODO: change the softmax layer -- no activate layer needed?
model.add(Dense(1)) # prediction should only be the current nesting level.
# model.add(Dense(out_dim, activation='softmax'))

# TODO: change the loss function 
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['acc'])

# 11 --------------------------------------------------------------------------

from keras.models import load_model
from controllers.mylstm_legacy import MYLSTM

if not CACHED:
    for i in range(N_EPOCHS):
        model.reset_states()
        history = model.fit(X_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=1,
                            verbose=1,
                            shuffle=False)
        model.save('models/parentheses_stateful_reg_mylstm')
else:
    model = load_model('models/parentheses_stateful_reg_mylstm',
                       custom_objects={'MYLSTM':MYLSTM})

# 12 --------------------------------------------------------------------------

model.reset_states()
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 13 --------------------------------------------------------------------------
# Generate samples
import generator as gen

print gen.complete_sentence_stateful('(1(2(3(', model, 64, char2int, int2char,BATCH_SIZE)
print gen.complete_sentence_stateful('((((4))', model, 64, char2int, int2char, BATCH_SIZE)
