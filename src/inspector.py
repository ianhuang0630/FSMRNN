import numpy as np
import features

def data_hash(feature_functions, sequence):
    return hash(str(feature_functions) + str(sequence))

class Inspector:
    def __init__(self, nn_config, offsets):
        """
            Input:
                - nn_config: a list that describes the layers of the nn.
                    format : [(layer_label, n_steps, n_neurons)]
                - offsets: a list that describes the offet for each timestep
                    format : [(layer_id, time_id) : offset]
            Output:
                nothing
        """
        self.states_struct = nn_config
        self.offsets = offsets
        self.attributions_scores = None
        self.attributions = None
        self.test_scores = None
        self.colnames = None


    # UTILS
    def timestep_neurons(self, i_layer, i_timestep):
        states_before = self.states_struct[:i_layer]
        n_before = sum(n * e for (_, n, e) in states_before)

        n_neurons = self.states_struct[i_layer][2]
        n_neurons_before = n_neurons * i_timestep

        first_index = n_before + n_neurons_before
        n_neurons = first_index + self.states_struct[i_layer][2]

        return range(first_index, n_neurons)

    def address_to_column(self, i_layer, i_timestep, i_neuron):
        first_neuron = self.timestep_neurons(i_layer, i_timestep)[0]
        return first_neuron + i_neuron

    def column_to_address(self, column):
        nn_config = self.states_struct
        layer_id = 0
        n_neurons_sofar = 0
        n_neurons_layer = nn_config[layer_id][1] * \
                                nn_config[layer_id][2]
        while n_neurons_sofar + n_neurons_layer <= column:
            n_neurons_sofar += n_neurons_layer
            layer_id += 1
            if not layer_id < len(nn_config):
                raise ValueError('Layer out of neural network')
            n_neurons_layer = nn_config[layer_id][1] * \
                                nn_config[layer_id][2]

        n_neurons_per_step = nn_config[layer_id][2]
        timestep_id = (column - n_neurons_sofar) // n_neurons_per_step
        if timestep_id >= nn_config[layer_id][1]:
            raise ValueError('Timestep out of neural network')
        neuron_id = (column - n_neurons_sofar) % n_neurons_per_step
        return layer_id, timestep_id, neuron_id

    @property
    def n_neurons(self):
        n = 0
        for _, steps, neurons in self.states_struct:
            n += steps * neurons
        return n


    # ATTRIBUTION
    def inspect(self, nn_states, feature_frame, score_obj):
        """
            Input:
                - nn_states : states
                - feature_frame : a Feature Frame object
                - score_obj : which score function to use

            Output: returns the score matrix:
                    rows: features functions
                    columns: each element
        """
        colnames, feature_matrix = feature_frame.data

        print 'Computing attribution scores'
        print 'Feture matrix dimensions:', feature_matrix.shape
        print 'States dimensions:', nn_states.shape

        n_features = feature_matrix.shape[1]
        n_states = nn_states.shape[1]

        out = np.empty((n_states,n_features))

        # For each feature...
        for i_f in range(n_features):
            print 'Computing score for feature ' + str(i_f) + ': ' + \
                         colnames[i_f]
            feature = feature_matrix[:,i_f]

            # For each layer...
            for i_layer, layer_struct in enumerate(self.states_struct):
                print 'Layer', i_layer
                (label_layer, n_steps, n_neurons) = layer_struct

                # For each cell...
                for i_timestep in range(n_steps):
                    print 'Timestep', i_timestep
                    offset = self.offsets[(i_layer, i_timestep)]

                    # Offsets the feature vector to match position in the NN
                    head_size = offset
                    tail_size = n_steps - (offset + 1)
                    f = feature[head_size:len(feature) - tail_size]

                    # Extracts the states
                    cell_indices = self.timestep_neurons(i_layer, i_timestep)
                    print 'Scoring neurons', cell_indices[0], 'to', cell_indices[-1]
                    S = nn_states[:,cell_indices]

                    # Does the maths
                    scores = score_obj.score_cell(f,S)

                    out[cell_indices,i_f] = scores

        self.attributions_scores = out
        self.attributions = None
        self.colnames = colnames
        return out, colnames

    def filter_attributions(self, filter_fun):
        if self.attributions_scores is None or self.colnames is None:
            raise ValueError('Need to score attributions first')

        # Gets the column IDs of attrobuted neurons
        col_attr = filter_fun(self.attributions_scores, self.colnames)

        # Converts to neuron address
        self.attributions = {}
        for feat, cols in col_attr.iteritems():
            self.attributions[feat] = [self.column_to_address(c) for c in cols]

        return self.attributions

    def not_attributed(self):
        if self.attributions is None or self.colnames is None:
            raise ValueError('Need to obtain attributions first')

        out = {}
        for feature, adresses in self.attributions.iteritems():
            attributed = set(self.address_to_column(*a) for a in adresses)

            timesteps = [(a[0],a[1]) for a in adresses]
            all_layer_neurons = set()
            for t in timesteps:
                all_layer_neurons |= set(self.timestep_neurons(*t))
            if len(all_layer_neurons) == 0:
                all_layer_neurons = set(range(self.n_neurons))

            not_att_cols = all_layer_neurons - attributed
            out[feature] = [self.column_to_address(c) for c in not_att_cols]

        return out

    @property
    def attributed_columns(self):
        if self.attributions is None or self.colnames is None:
            raise ValueError('Need to obtain attributions first')
        out = {}
        for feature, adresses in self.attributions.iteritems():
            out[feature] = [self.address_to_column(*a) for a in adresses]
        return out


    # TESTING
    def test(self, nn_states, feature_frame, attributions, score_obj, skip=[]):
        """
            Input
                - nn_states
                - feature_frame
                - attributions:
                    format: [feat_name : (layer_id, time_step_id, neuron_id)]
                - skip: prefix of features to skip
            Output: score results
        """

        # Generate all the features
        colnames, feature_matrix = feature_frame.data

        out = {}
        for feat_name, neuron_infos in attributions.iteritems():
            print 'Computing scores for', feat_name

            if any(pre in feat_name for pre in skip):
                print 'Skipped'
                continue

            # First, creates neuron activation matrix
            n_neurons = len(neuron_infos)
            n_symbols = feature_matrix.shape[0]
            neuron_mat = np.empty((n_symbols, n_neurons))

            lower_index_max = 0
            upper_index_min = feature_matrix.shape[0]

            for i_nid, neuron_address in enumerate(neuron_infos):
                layer_id, time_step_id, neuron_id = neuron_address
                activation_id = self.address_to_column(layer_id,
                                                    time_step_id,
                                                    neuron_id)

                n_steps = self.states_struct[layer_id][1]
                offset = self.offsets[(layer_id, time_step_id)]

                lower_index = offset
                tail_size = n_steps - (offset + 1)
                upper_index = feature_matrix.shape[0] - tail_size
                neuron_mat[lower_index:upper_index, i_nid] = \
                    nn_states[:,activation_id]

                if lower_index > lower_index_max:
                    lower_index_max = lower_index
                if upper_index < upper_index_min:
                    upper_index_min = upper_index

            neuron_mat = neuron_mat[lower_index_max:upper_index_min,]

            # Then state values
            f_col = colnames.index(feat_name)
            f = feature_matrix[lower_index_max:upper_index_min, f_col]

            # Does the maths
            out[feat_name] = score_obj.score(f,neuron_mat)

        self.test_scores = out
        return out



# VARIOUS SELECTORS
def filter_threshold_abs(threshold):
    def F(score_mat, colnames):
        out = {}
        for j, colname in enumerate(colnames):
            xceeds = score_mat[:,j] >= threshold
            neurons = np.where(xceeds)[0].tolist()
            out[colname] = neurons
        return out
    return F

def filter_threshold_percentile(ptile):
    def F(score_mat, colnames):
        thres = np.nanpercentile(score_mat.flatten(), ptile)
        print 'Filtering above threshold', thres
        out = {}
        for j, colname in enumerate(colnames):
            xceeds = score_mat[:,j] >= thres
            neurons = np.where(xceeds)[0].tolist()
            out[colname] = neurons
        return out
    return F

def filter_threshold_sddev(num_sigmas):
    def F(score_mat, colnames):
        p = score_mat.flatten()
        p = p[~(p == np.NaN)]
        mu = np.mean(p)
        sig = np.std(p)
        thres = mu + num_sigmas * sig
        print 'Filtering above threshold', thres
        out = {}
        for j, colname in enumerate(colnames):
            xceeds = score_mat[:,j] >= thres
            neurons = np.where(xceeds)[0].tolist()
            out[colname] = neurons
        return out
    return F