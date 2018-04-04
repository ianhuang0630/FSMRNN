import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import LSTM, Lambda
from keras import backend as K
from keras.models import load_model

from mylstm_legacy import MYLSTM, set_h

from collections import defaultdict
import random


def randomize_model(model, neurons, random_min=-1, random_max=1):
    # WARNING = ONLY WORKS WITH MYLSTM
    #         = ONLY SUPPORTS 1 TIME STEP PER LAYER
    layer_updates = defaultdict(dict)
    for l_id, t_id, n_id in neurons:
        layer_updates[l_id][n_id] = random.randrange(random_min, random_max+1)
    layer_updates = {k:v for k,v in layer_updates.iteritems()}
    hacked_model = set_neuron_value_mysltm(model, layer_updates)
    return hacked_model


def set_neuron_value_mysltm(model, layer_update_dict):
    """
        model: model which neurons are to be hacked
        layer_update_dict: dictionary with structure {layers:{neuron idx:new value}}
    """
    fname = 'model_bckup' + str(random.randint(1,10000000))
    model.save(fname) #, custom_objects={'MYLSTM':mylstm.MYLSTM})
    model2 = load_model(fname, custom_objects={'MYLSTM':MYLSTM})

    for layer_id, assignments in layer_update_dict.iteritems():
        class_name = model2.get_config()[layer_id]['class_name']
        if class_name != 'MYLSTM':
            raise ValueError('Layer to hack must be MYLSTM')
        for n_id, val in assignments.iteritems():
            set_h(model2.layers[layer_id], n_id, val)
    return model2


def set_neuron_value_lambda(model, layer_update_dict):
    """
        model: model which neurons are to be hacked
        layer_update_dict: dictionary with structure {layers:{neuron idx:new value}}
    """
    def controlit(x, updates):
        # x: input tensor
        # updates: dict of {neuron id: value}
        dim = x.get_shape().as_list()
        n_neurons = dim[1]
        mult_mask = [1]*n_neurons
        add_mask = [0]*n_neurons
        for pos_neuron, val_neuron in updates.iteritems():
            mult_mask[pos_neuron] = 0
            add_mask[pos_neuron] = val_neuron
        x = tf.multiply(x,mult_mask)
        x = tf.add(x,add_mask)
        return x

    new_output = None
    for i_layer, layer in enumerate(model.layers):
        if i_layer == 0:
            first_input = layer.input
            new_output  = layer.output
        else:
            new_output = layer(new_output)

        if i_layer in layer_update_dict:
            updates = layer_update_dict[i_layer]
            new_output = Lambda(controlit, arguments={'updates':updates})(new_output)

    return Model(inputs=first_input,outputs=new_output)