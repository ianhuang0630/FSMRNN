import numpy as np
from keras import backend as K
from keras.models import Model
from utils import preprocess

class Extractor:
    def __init__(self, model, layer_ids):
        print 'BEWARE _ ONLY SUPPORTS CONSECUTIVE LAYER IDS STARTING AT 0'
        self.model = model
        self.layer_ids = layer_ids

    def get_layer_index(self, layer_name):
        layers = [l['class_name'] for l in self.layer_ids]
        layer = layers.index(layer_name)
        return layer

    def get_layer_name(self, layer_id):
        return self.layer_ids[layer_id]['class_name']

    def unshuffle(self, states, batch_size):
        n_rows = states.shape[0]
        unshuffled_ix = preprocess.unshuffle_indices(n_rows, batch_size)
        states = states[unshuffled_ix,...]
        return states

    def run_for_layer(self, layer_name,
                            dataset,
                            batch_size=None,
                            unshuffle=False):
        lid = self.get_layer_index(layer_name)
        return get_states([lid], dataset, batch_size, unshuffle)

    def get_states(self, dataset,
                        batch_size=None,
                        unshuffle=False):
        print 'Creates spy models'
        spy_models = []
        for l_id in self.layer_ids:
            print '... for id', l_id, ':', self.model.layers[l_id]
            m = Model(inputs = self.model.input,
                    outputs = self.model.layers[l_id].output)
            spy_models.append(m)

        print 'Gets the activations for the hidden states'
        self.model.reset_states()

        def batch_predict(batch):
            layer_states = []
            for spy in spy_models:
                S = spy.predict(batch)
                if S.ndim == 3:
                    new_dims = (S.shape[0], S.shape[1]*S.shape[2])
                    S = np.reshape(S, new_dims)
                layer_states.append(S)
            states = np.concatenate(layer_states, axis=1)
            if states.ndim != 2 or states.shape[0] != batch.shape[0]:
                raise ValueError('Something went wrong in state extraction')
            return states

        if batch_size is None:
            states = batch_predict(dataset)

        else:
            i_start = 0
            batch_states = []
            while i_start + batch_size - 1 < dataset.shape[0]:
                batch = dataset[i_start:(i_start + batch_size),...]
                s = batch_predict(batch)
                batch_states.append(s)
                i_start += batch_size
            states = np.concatenate(batch_states)

        if unshuffle:
            states = self.unshuffle(states, batch_size)

        return states

    def get_structure(self):
        print 'Gets structure'
        struct = []
        for l_id in self.layer_ids:
            name = str(self.model.layers[l_id])
            dims = self.model.layers[l_id].output_shape
            if len(dims) == 2:
                n_steps, n_neurons = 1, dims[1]
            elif len(dims) == 3:
                n_steps, n_neurons = dims[1], dims[2]
            else:
                raise ValueError('Cannot deal with layers of shape', dims)
            struct.append((name, n_steps, n_neurons))
        return struct

    def get_offets(self):
        print 'Gets offets'
        print 'WARNING +++ NOT SUITABLE FOR NON_FORWARD LAYERS'
        struct = self.get_structure()
        offets = {}
        last_offset = 0
        cur_l = 0
        cut_t = 0
        for i,lid in enumerate(self.layer_ids):
            n_steps = struct[i][1]
            if n_steps == 1:
                offets[(i,0)] = last_offset
            elif n_steps > 1:
                for t in range(n_steps):
                    offets[(i,t)] = t
                last_offset = n_steps - 1
        return offets
