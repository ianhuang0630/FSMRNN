from custommodels import my_recurrent_models
from custommodels import recurrent_models

import keras.backend as K

import numpy as np
from shutil import copy
import unicodedata
import pickle
import random
import h5py
import nltk
import sys
import os
import re


#################################################
# General function to load models and data sets #
#################################################
def load_setup(num_layer,
                path,
                model_weights,
                model_settings_name ,
                new_batch_size,
                new_size,
                new_time_skip,
                is_stateful):

    # Backup of the weights + weird stuff
    new_model_weights = model_weights + ".bak"
    copy(model_weights, new_model_weights)

    fp = h5py.File(new_model_weights, 'r+')
    for lstm_name in fp.attrs['layer_names'][:num_layer]:
        num = fp[lstm_name][lstm_name]['bias'].size/4
        print(num)
        print(lstm_name)
        fp.create_dataset("%s/%s/flag" % (lstm_name, lstm_name),
                data = np.zeros((num,)))
        fp.create_dataset("%s/%s/value" % (lstm_name, lstm_name),
                data = np.zeros((num,)))
        x = np.concatenate((
            fp[lstm_name].attrs['weight_names'],
            [lstm_name + '/flag',lstm_name + '/value']
        ))
        fp[lstm_name].attrs['weight_names'] = x
    fp.close()


    # Loads the setting files
    with open(model_settings_name, 'rb') as f:
        n_epochs, n_checkpoint, max_vocab_size, time_skip, sequence_length, \
        sequences_per_batch, batch_size, model_architecture, model_type,    \
        architecture_type, parameter_updates = pickle.load(f)
    if new_size is not None:
        sequence_length = new_size
    if time_skip is not None:
        time_skip = new_time_skip
    if new_batch_size is not None:
        batch_size = new_batch_size

    # Loads the model
    model, inverting_dict, identity_dict, model_type =\
     my_model_loader(path, new_model_weights, model_settings_name,
                    new_batch_size = new_batch_size, new_size = new_size,
                    is_stateful = is_stateful)

    return model, inverting_dict, identity_dict, model_type


#################
# Model loaders #
#################
def my_model_loader(path, model_weights, model_settings_name,
                     new_size = None, is_stateful = True,
                     new_batch_size = None):
    path_basic_name = path.replace('/','_')

    print 'Loading model', model_settings_name

    with open(model_settings_name) as f:
        n_epochs, n_checkpoint, max_vocab_size, \
        time_skip, sequence_length, sequences_per_batch, \
        batch_size, model_architecture, model_type, \
        architecture_type, parameter_updates = pickle.load(f)

    print 'type:', model_architecture

    name_extension = ''
    for i, j in parameter_updates.iteritems():
        name_extension = name_extension + '_' + i + '_' + str(j)

    # Loads dictionaries
    with open('dicts/' + path_basic_name[:-4] + '_architecture_' + \
                 architecture_type + '_type_' + model_type + name_extension + \
                  '_inverting_dict.pickle', 'rb') as f:
        inverting_dict = pickle.load(f)
    with open('dicts/' + path_basic_name[:-4] + '_architecture_' + \
                 architecture_type + '_type_' + model_type + name_extension + \
                  '_identity_dict.pickle', 'rb') as f:
        identity_dict = pickle.load(f)
    filtered_vocab = list(identity_dict.keys())

    # Loads model
    model = None
    if new_size is not None:
        model = getattr(my_recurrent_models, model_architecture)\
                (filtered_vocab, new_batch_size, new_size, is_stateful = True,\
                 parameter_updates = parameter_updates)
    else:
        model = getattr(my_recurrent_models, model_architecture)\
                (filtered_vocab, batch_size, sequence_length, is_stateful = True,\
                 parameter_updates = parameter_updates)
    model.compile(loss='categorical_crossentropy', optimizer='nadam',\
                     metrics = ['accuracy'])
    model.load_weights(model_weights)

    return model, inverting_dict, identity_dict, model_type


##################
# Corpus loaders #
##################
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', unicode(input_str, 'latin-1'))
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def get_corpus(path, max_vocab_size=-1, model_type = 'char'):
    print 'Reading file', path
    if model_type == 'word':
        text = remove_accents(open(path).read()).lower()
    else:
        text = open(path).read().lower()
    print 'Number of Words in Current Corpus:', len(text)

    chars = set(text)
    if model_type == 'word':
        dataset = re.findall(r'\S+|\n',text)
        dataset = filter(lambda w: not w in [' '], dataset)

    elif model_type == 'char':
        dataset = list(text)

    else:
        dataset = list(text.encode("hex"))

    freq_dist = nltk.FreqDist(dataset)
    if max_vocab_size>-1:
        return filter(lambda w: not w in freq_dist.keys()[max_vocab_size:], dataset)
    else:
        return dataset