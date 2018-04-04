
# coding: utf-8

# # Libraries, Headers and Stuff

# In[1]:


import sys
sys.path.append('../../src')


# In[2]:


CACHED = False
CONTINUE = True

N_STATES = 96
N_EPOCHS =  50
BATCH_SIZE = 32

MAX_VOCAB_SIZE = -1
MAX_SEQUENCE_SIZE = -1

LAYERS_TO_CHECK = [0]

import scores
METRIC = scores.Correlation()

SAMPLES_VERIFY = 8
TEST_RATIO = .1

MODEL_NAME = '4_layers_96_cells'


# In[3]:


import datetime
import os

FOLDER_OUT = MODEL_NAME + '-' + str(datetime.datetime.today())[:16]
FOLDER_OUT = FOLDER_OUT.replace(' ','_')
if not os.path.exists(FOLDER_OUT):
    os.makedirs(FOLDER_OUT)

print FOLDER_OUT


# In[4]:


import numpy as np
import pandas as pd
import keras


# In[5]:




# In[6]:


import random
random.seed(55555)


# # Creates dataset

# In[7]:


import kernelhelpers
reload(kernelhelpers)

import pickle

num_layer = 1

path = 'corpus/linux_kernel_val.txt'
model_weights = 'models/corpus_linux_kernel_val_architecture_stateful_type_char_layers_1_hidden_units_128_epoch_49_weights.h5'
model_settings_name = 'settings/corpus_linux_kernel_val_architecture_stateful_type_char_layers_1_hidden_units_128_settings.pickle'
new_batch_size = BATCH_SIZE
new_size = 1
new_time_skip = 1
is_stateful = True

# Loads the model
old_model,char2int,int2char,model_type =     kernelhelpers.load_setup(num_layer,
                path,
                model_weights,
                model_settings_name ,
                new_batch_size,
                new_size,
                new_time_skip,
                is_stateful)

raw_data = kernelhelpers.get_corpus('corpus/linux_input.txt')

dict_location = 'dicts/' + MODEL_NAME
with open(dict_location + '_char2int.pickle' , 'wb') as f:
    pickle.dump(char2int, f)
with open(dict_location + '_int2char.pickle' , 'wb') as f:
    pickle.dump(int2char, f)

print 'Number of charaters', len(raw_data)
print 'Number of charaters', len(set(raw_data))


# In[8]:


import utils.preprocess as pre

# Generates data
full_size = len(raw_data)
train_size = int((1-TEST_RATIO) * full_size)

train_sequence = raw_data[:train_size]
test_sequence = raw_data[train_size:]

raw_train_sequence = [char2int[c] for c in train_sequence]
raw_test_sequence  = [char2int[c] for c in test_sequence]

# If necessary, truncates:
if MAX_SEQUENCE_SIZE > 0:
    train_size = int(MAX_SEQUENCE_SIZE * (1-TEST_RATIO))
    test_size = int(MAX_SEQUENCE_SIZE * TEST_RATIO)
    raw_train_sequence = raw_train_sequence[:train_size]
    raw_test_sequence  = raw_test_sequence[:test_size]

print 'Training smybols:', len(raw_train_sequence)
print 'Testing smybols:', len(raw_test_sequence)
print 'Size vocabulay:', len(char2int)


# In[9]:


import numpy as np
import utils.preprocess as pre
reload(pre)

# Preprocesses training
# SPLITS
X_train_raw = raw_train_sequence[:-1]
y_train_raw = raw_train_sequence[1:]
# ENCODES
X_train = pre.one_hot_encode_seq(X_train_raw, n_chars=len(int2char))
y_train = pre.one_hot_encode_seq(y_train_raw, n_chars=len(int2char))
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
X_test = pre.one_hot_encode_seq(X_test_raw, n_chars=len(int2char))
y_test = pre.one_hot_encode_seq(y_test_raw, n_chars=len(int2char))
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


# # Builds model

# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,SimpleRNN
from keras.layers import Lambda
from keras import regularizers
from keras.optimizers import RMSprop,Adam

from controllers.mylstm_legacy import MYLSTM

in_dim = X_train.shape[1:]
out_dim = y_train.shape[1]

model = Sequential()
model.add(MYLSTM(N_STATES,
                 stateful=True,
                 batch_size=BATCH_SIZE,
                 input_shape=in_dim,
                 return_sequences=True))
model.add(MYLSTM(N_STATES,
                 stateful=True,
                 return_sequences=True))
model.add(MYLSTM(N_STATES,
                 stateful=True,
                 return_sequences=True))
model.add(MYLSTM(N_STATES,
                 stateful=True))
model.add(Dense(out_dim, activation='softmax'))

optimizer = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# In[11]:


from keras.models import load_model
from controllers.mylstm_legacy import MYLSTM
from keras.callbacks import ModelCheckpoint

if not CACHED:
    if CONTINUE:
        model = load_model('thib-models/' + MODEL_NAME + '/' + MODEL_NAME, custom_objects={'MYLSTM' :MYLSTM})
    model.reset_states()
    checkpoint = ModelCheckpoint(FOLDER_OUT + '/model-{epoch:02d}-{acc:.2f}',
                                 monitor='val_loss')
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=N_EPOCHS,
                        verbose=1,
                        shuffle=False,
                        callbacks = [checkpoint])
    model.save('models/' + MODEL_NAME )
    model.save('models/' + MODEL_NAME + '.bckup')
else:
    model = load_model('models/' + MODEL_NAME, custom_objects={'MYLSTM' :MYLSTM})


# # Evaluate

# In[12]:


model.reset_states()
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[13]:


# Generate samples
import generator as gen
reload(gen)

print gen.complete_sentence_stateful('if ', model, 256, char2int, int2char, BATCH_SIZE)
print '*****'
print gen.complete_sentence_stateful('else', model, 256, char2int, int2char, BATCH_SIZE)
