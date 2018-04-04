import numpy as np
from collections import deque

from sklearn.cluster import KMeans
from keras.models import load_model
from controllers.mylstm_legacy import MYLSTM
import extractor

import multiprocessing

import pickle

from sklearn.metrics import silhouette_score



BATCH_SIZE = 10
MODEL_FILE = "../experiments/boolean/models/boolean.h5"
TESTDATA_FILE = "../experiments/boolean/test_data_boolean.pkl" 


class FSMExtractor:

    def __init__(self, model, data, quantization, transition, minimization):
        """ Class for extracting the DFA from an RNN
        Input:
            model (keras.models):
            data (Tuple): containing the inputs and ground truth.

            quantization (func): function for quantization
            transition (func):  functino for state transition extraction
            minimization (func): function for DFA minimization

        """

        self.model = model
        self.input = data[0]
        self.ground_truth = data[1]

        # the different functions involved      
        self.quant_func = quantization
        self.trans_func = transition
        self.minim_func = minimization

    
    def get_hidden_states(self):
        # TODO: documentation
        """
        
        """ 

         ## reset rnn states
        self.model.reset_states()

        ## predict test set
        labels = self.model.predict(self.input, batch_size = BATCH_SIZE)
        
        ## extract states for all sequences accross all chars.
        ex = extractor.Extractor(self.model, [0])
        states = ex.get_states(self.input, batch_size=BATCH_SIZE, \
                unshuffle=True)
        states_perchar = states.reshape(self.input.shape[0]*self.input.shape[1],-1) 
        return states_perchar

    def extract(self):
        
        states_perchar = self.get_hidden_states()

        ## format of states:
        # for input1: t1_s1, t1_s2, t1_s3... t1_s16, t2_s1, t2_s2, t2_s3 ...
        # for input2: t1_s1, ....
        # ...
        ##

        return self.cluster_transition(states_perchar)


    def cluster_transition(self, states_perchar):
        # XXX: every 50 rows of states_perchar is an example, and each row corresponds
        # to a letter, and contains the activations of 16 neurons.
        
        ## clustering using self.quant_func
        cluster_perchar = self.quant_func.quantize(states_perchar)

        num_examples = self.input.shape[0]
        num_letters = self.input.shape[1]
        num_hid_states = states_perchar.shape[1]

        ## extracting DFA state transitions using self.trans_func for every test example
        trans_per_model = []

        # TODO: randomize this
        example_indices = np.random.choice(num_examples, 100)
        for i in example_indices: 
            this_input = self.input[i,:,:]
            this_clusters = cluster_perchar[i*num_letters:(i+1)*num_letters]
            
            trans_per_model.append(self.trans_func.extract_transitions(self.model, 
                                                        this_input, this_clusters))
        
        
        aggregate = {}

        for trans_dic in trans_per_model:
            for item in trans_dic:
                if item in aggregate:
                    aggregate[item] += trans_dic[item]
                else:
                    aggregate[item] = trans_dic[item]

        

        # turn transition counts into probabilities: P(state2 | state1, transition)
        # TODO

        sums = {}

        for key in aggregate:
            firsttwo = (key[0], key[1])
            if firsttwo not in sums:
                sums[firsttwo] = aggregate[key]

            else:
                sums[firsttwo] += aggregate[key]


        for key in aggregate:
            firsttwo = (key[0], key[1])
            aggregate[key] /= float(sums[firsttwo])


        return aggregate


class kMeansQuantizer: 
    def __init__(self, k, max_iter=300, n_init=10, random_seed=None):
        """
        Inputs:
            k (int): number of clusters
            k ()
        """
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_seed

        self.kmeans_model = KMeans(self.k, max_iter=self.max_iter, 
                n_init=self.n_init, random_state=self.random_state)

    def quantize(self, states):
        """
        Inputs:
            states (np.array): array of size 
                (total # of characters, state_dimension). 
        Outputs:
            cluster_label (np.array): 1-dimensional array, of length
                'total # of characters', with labeling in range(self.k)
        """
        self.kmeans_model.fit(states)
        return self.kmeans_model.labels_

    def getSilhouetteCoef(self, states, random_seed=None):
        """
        Returns the silhouette coefficient
        """

        # shrinking labels and states arrays to reduce computational cost

        if states.shape[0] > 1000:
            
            chooseidx = np.random.choice(states.shape[0], 1000)
            a = states[chooseidx]
            b = self.kmeans_model.labels_[chooseidx]

        else:
            a = states
            b = self.kmeans_model.labels_
        
        return silhouette_score(a,b)


    def getInertia(self):
        """
        Returns the inertia 
        """
        pass
    def getCentroids(self):
        """
        Returns the centers of the centroids
        """

        return self.kmeans_model.cluster_centers_

    def getLabels(self):
        """
        This function allows macrostate predictions to be returned without 
        having to run Kmeans again
        """
        return self.kmeans_model.labels_


class SampleBasedTransExtractor:
    def  __init__(self, sampling_proportion=0.2, random_seed=None, conv_func=None):
        """
        Inputs:
            sampling_proportion (float, optional): The fraction of
                data sampled for transitions
            random_seed (int, optional): Random seed to make sampling
                deterministic if needed, for testing purposes.
        """

        self.sample_fraction = sampling_proportion
        self.random_seed = random_seed
        self.conv_func = conv_func

    def extract_transitions(self, model, data, state_predictions):

        """
        Inputs:
            model (keras.Model): model of interest
            data (np.array): data being sampled. 
            state_predictions (np.array): the DFA states that every
                word in data maps to
        
        Returns:
            transitions (dict): the keys are tuples of the form
                ([START_STATE], [TRANSITION], [END_STATE]) and the
                values are the counts normalized across all the 
                transitions to any END_STATE given START_STATE and 
                TRANSITION.

                e.g.
                {
                    (0, '0', 0): 0.2
                    (0, '1', 0): 0.1
                    (0, '0', 1): 0.8
                    (0, '1', 1): 0.9
                    (1, '0', 0): 0.4
                    (1, '1', 1): 0.0 
                    (1, '0', 1): 0.6
                    (1, '1', 0): 1.0
                }

        """
        # find the first occurence of every state
        occ = self.find_occurences(state_predictions)
        
        # insert every possible letter from the alphabet
        
        states = set(state_predictions)

        transitions = {}
        # for every state:
        for i in states:
            # find occurence
            for index in occ[i]:
                if index < data.shape[0]-1:
                    current_symbol = self.conv_func(data[index])
                    next_symbol = self.conv_func(data[index+1])
        
                    # finding the states of next symbol
                    next_state = state_predictions[index+1]
        
                    # increment 1 to the count for the tuple (i, next_symbol, next_state)
                    if (i, next_symbol, next_state) not in transitions:
                        transitions[(i,next_symbol, next_state)] = 1
                    else:
                        transitions[(i,next_symbol, next_state)] += 1
        
        # we are letting -1 be the initial state:
        # getting all state transitions from the initial state
       
        # iterating through all of the sequences, get first character as a transtion and the first state

        #import ipdb; ipdb.set_trace() 
        ## TODO: finish this
        #transitions[(-1, self.conf_func(data[0]), )] 

        return transitions

    def find_occurences(self, state_predictions):
        """ Helper function to find the occurence of macostates in a given sequence

        Input:
            state_predictions (list): state predictions, should be a list of 
                integers.
        
        Output:
            occ (dict): For every single class, a list of indices in state_predictions

        """
        
        occ = {i:[] for i in list(set(state_predictions))}
        
        for (idx, item) in enumerate(state_predictions):
            occ[item].append(idx)
            
        return occ
        

if __name__ == "__main__":
    # certain set of tests here
    
    model = load_model(MODEL_FILE)
    alphabet = set(["X", "&", "|", "1", "0"])

    def arr2char(a):
        arr = a.tolist()
      	if arr == [1,0,0,0,0]:
            return "x"
	if arr == [0,1,0,0,0]:
	    return "&"
	if arr == [0,0,1,0,0]:
            return "0"
	if arr == [0,0,0,1,0]:
	    return "1"
	if arr == [0,0,0,0,1]:
	    return "|"
	else:
            raise ValueError("sorry, can't recognize label")
    
    test_data = pickle.load(open(TESTDATA_FILE))
        
    quantizer = kMeansQuantizer(5)
    transition_extractor = SampleBasedTransExtractor(conv_func=arr2char)    
    # setting up the FSM extraction    
    # TODO: improve minimization algoirthm -- right now set to NONe
    experiment = FSMExtractor(model, test_data, quantization=quantizer, transition=transition_extractor, minimization=None)
    fsm = experiment.extract()
    print(fsm)
    
    import matplotlib.pyplot as plt
    prob = fsm.values()
    
    
    x = np.arange(len(fsm.keys()))
    
    def to_str_names(b):
        
        list_names = [] 
        for a in b:
            list_names.append(str(a[0]) +"\n" + str(a[1]) + "\n" + str(a[2]) + "\n")
        
        return tuple(list_names)

    # x_names = set([str(i) for i in fsm.keys()])
    x_names = to_str_names(fsm.keys())

    plt.bar(x, prob)
    plt.xticks(x, x_names)
    plt.show()





