from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import ttest_ind
import numpy as np
import keras.backend as B

from collections import defaultdict
import random

import controllers.mylstm_legacy
import features
from utils import preprocess

from controllers import control


def flatten_dict_array(dicts):
    keys = [d.keys() for d in dicts]
    concat_keys = reduce(lambda x,y: x+y, keys)
    unique_keys = set(concat_keys)

    out = {}
    for k in unique_keys:
        out[k] = [d[k] for d in dicts if k in d]
    return out


class Verifier:

    def __init__(self, model, attrib_neurons, not_attrib_neurons):
        self.model = model
        self.attrib_neurons = attrib_neurons
        self.not_attrib_neurons = not_attrib_neurons
        self.prediction_scores = None

    def run(self, y_feat_frame, X, y,
               shuf_batch_size = None,
               sample_size = 25,
               random_min=-1,
               random_max=1,
               one_hot = True):
        """
            BEWARE: ONLY WORKS WITM MYLSTM!
        """
        def stratified_acc(model):
            self.model.reset_states()
            if shuf_batch_size is not None:
                pred = model.predict(X, batch_size=shuf_batch_size, verbose=0)
            else:
                pred = model.predict(X, verbose=0)

            scores = {}
            for y_ff_val in y_ff_values:

                sub_y = y[y_ff == y_ff_val, ...]
                sub_pred = pred[y_ff == y_ff_val, ...]

                if one_hot:
                    sub_y = np.argmax(sub_y, axis=1)
                    sub_pred = np.argmax(sub_pred, axis=1)
                    f1_avg = 'macro'
                else:
                    f1_avg = 'binary'

                scores[y_ff_val] = f1_score(sub_y, sub_pred, average=f1_avg)

            return scores

        if not X.shape[0] == y.shape[0]:
            raise ValueError('Incorrect number of rows')

        # Run the feat function on labels
        ff_names, ff_val = y_feat_frame.data

        # For every feature...
        all_scores = {}
        for ff_index, ff_name in enumerate(ff_names):

            print '*** Testing the neurons for feature', ff_name

            # Extracts vector and distinct values
            y_ff = ff_val[:,ff_index]
            y_ff_values = list(set(y_ff.tolist()))

            # shuffles the lables to match the order in y
            if shuf_batch_size is not None:
                y_ff_idx = preprocess.shuffle_indices(ff_val.shape[0],
                                                      shuf_batch_size)
                y_ff = y_ff[y_ff_idx]

            # Gets the neurons and baseline neurons
            neurons = self.attrib_neurons[ff_name]
            baseline_neurons = self.not_attrib_neurons[ff_name]

            # Gets original predictions
            print '* Computing original accuracy'
            scores = [stratified_acc(self.model)]
            orig_scores = flatten_dict_array(scores)

            # Hacks the baseline neurons
            print '* Computing baseline accuracies'
            scores = []
            for i in range(sample_size):
                print 'Round', i
                if len(baseline_neurons) > len(neurons):
                    to_hack = random.sample(baseline_neurons, len(neurons))
                else:
                    to_hack = baseline_neurons
                hacked_model = control.randomize_model(self.model, to_hack)
                acc = stratified_acc(hacked_model)
                scores.append(acc)
            base_scores = flatten_dict_array(scores)

            # Hacks the candidate neurons
            print '* Computing candidate accuracies'
            scores = []
            for i in range(sample_size):
                print 'Round', i
                hacked_model = control.randomize_model(self.model, neurons)
                acc = stratified_acc(hacked_model)
                scores.append(acc)
            cand_scores = flatten_dict_array(scores)

            all_scores[ff_name] = {
                'original': orig_scores,
                'candidate': cand_scores,
                'baseline': base_scores
            }

            print all_scores[ff_name]

            self.prediction_scores = all_scores

        return all_scores


    def test_diff(self):
        if self.prediction_scores is None:
            raise ValueError('Must compute scores first!')
        test_scores = {}
        for feat_name, feat_scores in self.prediction_scores.iteritems():
            print 'Testing for feature', feat_name
            x1 = feat_scores['candidate']
            x2 = feat_scores['baseline']
            for val in x1:
                test_res = ttest_ind(x1[val], x2[val], equal_var=False)
                print 'Value', val, ':', test_res
                test_scores[(feat_name, val)] = test_res
        return test_scores
