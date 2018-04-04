import numpy as np
from collections import deque

from sklearn.cluster import KMeans
from keras.models import load_model
from controllers.mylstm_legacy import MYLSTM
import extractor

import pickle

# pull boolean. jupyter and pickl enew testData

BATCH_SIZE = 10
MODEL_FILE = "../experiments/boolean/models/boolean.h5"
TESTDATA_FILE = "../experiments/boolean/test_data_boolean.pkl" # TODO: add path to pickle file
K_NUM = 5

class FSMExtractor:
    def __init__(self, model, test_data, arr2char, alphabet, k_num=3):
        """
        Inputs:
            model:
            test_data (list):
            alphabet (set):
            k_num (int):

        Returns:
            dfa (DFA): 
        """

        self.rnn = model
        self.data = test_data[0] # test data to extract features
        self.data_gt = test_data[1]
        self.alphabet = alphabet
        self.k_num = k_num 
        self.arr2char = arr2char


    def extract(self):
        ## extract states from the model trained through the test set
        self.rnn.reset_states()
        print("predicting labels...")

        ## predictions to label states positive and negative
        labels = self.rnn.predict(self.data, batch_size=BATCH_SIZE)

        ## extract hidden states for all sequences across all chars.
        ex = extractor.Extractor(self.rnn, [0])
        states = ex.get_states(self.data, batch_size=BATCH_SIZE, \
                unshuffle=True)
        states_perchar = states.reshape(self.data.shape[0]*self.data.shape[1],-1)

        ## clustering using k-means
        print("Clustering...")
        kmeans = KMeans(n_clusters=self.k_num, random_state=0)
        kmeans.fit(states_perchar)
        print("Complete.")
        print("Inertia: {}".format(kmeans.inertia_))



        #######################################################################
        # search for first occurrence of each state in kmeans.labels_
        # TODO: randomize, and make it pass thorugh all examples 
        first_occur = self.find_first(kmeans.labels_)
        # indices correspond to the index in 250000,16, change first_occur to seq_num and char_num
        first_seq_char = {element: [first_occur[element]//50, first_occur[element]%50] for element in first_occur}
        #######################################################################

        ## create state objects and visited/unvisited markers
        states = [State("s{}".format(i)) for i in range(self.k_num)]
        visited = [False]*self.k_num

        ## initializing variables for BFS
        current_state_num = 0
        visited[0] = True
        q = deque()
        q.append(states[0])

        # create DFA object
        dfa = DFA()
        dfa.set_omega(self.alphabet)
        dfa.set_Q(states)
        dfa.set_q_0(states[0])

        prev_input = None
        counter = 0; threshold = 100

        while q:

            print("Iteration # {}".format(counter))
            print("------------------------------")

            counter += 1
            if counter > threshold:
                raise Warning("while loop doesn't seem to exit")

            current_state = q.popleft()
            current_state_num = current_state.get_state_num()

            # label of the timestep is
            label_state = labels[first_occur[current_state_num]]
            # TODO: label_state is a double -- may have to round?

            if label_state == 1:
                dfa.add_F(current_state)

            # locate the first occurrence of current_state

            seq_num = first_seq_char[current_state_num][0]
            char_num = first_seq_char[current_state_num][1]
            sofar_str = self.data[seq_num][:char_num+1]

            # sofar_str should lead to the state being observed
            
            # this is naive -- because of padding.
            prev_input = sofar_str[-1] # this is the last letter in the previous sequence

            print("prev character = {}".format(self.arr2char(prev_input.tolist())))

            for i in self.next_alphabet(prev_input):

                print("Trying character {}".format(self.arr2char(i.tolist())))

                # add i at the back of the test sequence
                new = np.vstack((sofar_str, i))
                len_string = len(i)

                # new will not be 50 in length anymore. This can be remedied by 
                # deleting the first len(i) elements, given that they are X's
                
                if new.shape[0] > 50:
                    if any([ all(element == np.array([1,0,0,0,0])) for element in new[:len_string]]):
                        new = new[len_string:]
                    else:
                        raise ValueError("The sequence doesn't start with enough X's")
                elif new.shape[0] < 50:
                    insert = np.tile(np.array([1,0,0,0,0]), (50 - new.shape[0],1))
                    new = np.vstack((insert, new))

                # ok... so this part is super inefficient, but i think it works.
                train_copy = self.data[seq_num:seq_num + BATCH_SIZE].copy()
                train_copy[0] = new

                # plug sequence into RNN model
                new_states = ex.get_states(train_copy, 
                        batch_size=BATCH_SIZE, unshuffle=True)
                
                # feature extraction at position first_occurrance +1 
                
                next_s = new_states[0,-16:]

                # kmeans classification

                next_state_num = kmeans.predict(next_s)[0] # next_state is an int
                next_state = states[next_state_num]

                # if the next state has already been visited, add_delt, but
                # don't enqueue.

                trigger_char = self.arr2char(i.tolist())

                current_state.add_delt(trigger_char, next_state) 
                # linking current_state to a State object
                dfa.add_delta(current_state, trigger_char, next_state)

                if not visited[next_state_num]:
                    # enqueue
                    q.append(next_state)

            # process of adding to DFA object

        return dfa, states

        # return a feature function, which gives state at every time step

    def next_alphabet(self,prev_input):
        """
        Input:
            prev_input: 1 dimensional numpy array holding the category of the 
                previous input.

        returns:
            a list of numpy arrays. 

        returns a subset of self.alphabet fitting for the timestep.
        """
        # XXX: this is experiment specific

        # {'1': 3, '0': 2, '|': 4, 'X': 0, '&': 1}

        if prev_input == None: 
            return [np.array([1,0,0,0,0])] # X

        elif np.array_equal(prev_input, np.array([1,0,0,0,0])): # if "X"
            return [np.array([1,0,0,0,0]), 
                    np.array([0,0,0,1,0]), 
                    np.array([0,0,1,0,0])] # X, 1, 0

        elif np.array_equal(prev_input, np.array([0,1,0,0,0])) or \
            np.array_equal(prev_input,np.array([0,0,0,0,1])):
            return [np.array([0,0,0,1,0]),
                    np.array([0,0,1,0,0])] # 0 or 1

        elif np.array_equal(prev_input, np.array([0,0,0,1,0])) or \
            np.array_equal(prev_input, np.array([0,0,1,0,0])): # 1 or 0         

            return [np.array([0,1,0,0,0]), # &
                    np.array([0,0,0,0,1])] # |
                    
        else:
            raise ValueError("Input does not belong to [X, 1, 0, &, |]")

    def find_first(self, labperstep):
        """
        Helper function for extract.
        
        Inputs:
            labperstep (list): the 
        
        Returns:
            first_occur: a dictionary where the keys corresponds to the 
        """

        first_occur = {}

        for (idx, i) in enumerate(labperstep):
            if i not in first_occur:
                first_occur[i] = idx

        assert len(first_occur) == self.k_num, \
             "Length of dictionary doesn't match the number of clusters."

        return first_occur
    

class DFA:
    def __init__(self, Q=None, omega=None, q_0=None):
        """
        Inputs:
            Q (list): the finite set of states
            omega (list [string]): a finite, nonempty input alphabet
            delta (dict): a series of transition functions
            q_0 (State): the starting state
            F (set): the set of accepting values (?? Needed?)
        """

        self.Q = Q
        self.omega = omega
        self.delta = {}
        self.q_0 = q_0
        self.F = {}

    def set_Q(self, Q):
        self.Q = Q

    def add_Q(self, Q):
        """
        Adding a state in to the set Q
        """
        if self.Q == None:
            self.Q = set([Q])
        else:
            self.Q.add(Q)

    def set_omega(self, omega):
        self.omega = omega

    def initiate_delta(self):
        """
        Initialize self.delta to hold keys linking every possible state1
        to every possible state2 for every possible transition
        """

        # initiate delta given self.Q (set of states) and self.omega (alphabet)

        for state1 in self.Q:
            for transition in self.omega:
                for state2 in self.Q:
                    # initializing all values to None for now.
                    self.delta [(state1, transition, state2)] = None


    def add_delta(self, state1, transition, state2):
        # dictionary mapping (state1, transition) -> state2

        # delta should have all possible transition in the keys of the dictionary
        # e.g. (state1, transition, state2)

        if (state1, transition) not in self.delta:
            self.delta[(state1, transition)] = state2
        else:
            print("state {} and transition {} are already in keys of delta."\
                .format(state1, transition))

    def set_q_0(self, q_0):
        self.q_0 = q_0

    def add_F(self, F):
        # TODO
        if self.F == None:
            self.F = set([F])

        else:
            self.F.add(F)

    def __str__(self):
        """
        Some way to print out the states
        """
        return "Q: {} \n omega: {} \n delta: {} \n q_0: {} \n F: {}\n".\
            format(self.Q, self.omega, self.delta, self.q_0, self.F)

class State(object):
    def __init__(self, name):
        """
        Inputs:
            name (string): Name of the state.
        """
        self.name = name
        self.delta = {}

    def add_delt(self, trigger, next_state):
        """
        Change the delta function associated with the state
        """
        if trigger not in self.delta:
            self.delta[trigger] = next_state

        else:
            if next_state != self.delta[trigger]:
                raise ValueError("Contradiction discovered in delta function")
            else: 
                pass # nothing needs to be changed

    def get_state_num(self):
        # if self.name has "s" at the beginning

        if self.name[0] == "s":
            return int(self.name[1:])
        else:
            raise ValueError("Can't return a state number for state name {}"\
                    .format(self.name))

    def __eq__(self, other):
        if other.name == self.name:
            return True
        else:
            return False

    def __str__(self):
        return self.name



if __name__ == "__main__":
# def main():
    """
    model_filename (string): path to the model
    """
    model = load_model(MODEL_FILE)
    alphabet = set(["X", "&", "|", "1", "0"])

    def arr2char(arr):
        if arr == [1,0,0,0,0]:
            return "X"
        if arr == [0,1,0,0,0]:
            return "&"

        if arr == [0,0,1,0,0]:
            return "0"

        if arr == [0,0,0,1,0]:
            return "1"

        if arr == [0,0,0,0,1]:
            return "|"
        else:
            raise ValueError("Sorry, can't recognize label")

    test_data = pickle.load(open(TESTDATA_FILE))

    fsm_ex = FSMExtractor(model, test_data, arr2char, alphabet, k_num=K_NUM)

    dfa, states = fsm_ex.extract()
    print(dfa)

# if __name__ == "__main__":

#   main()

