import numpy as np
from collections import deque

from sklearn.cluster import KMeans
from keras.models import load_model
from controllers.mylstm_legacy import MYLSTM
import extractor

import pickle

# pull boolean. jupyter and pickl enew testData

# CACHED = "../experiments/boolean/dfa_object1.pkl"
CACHED = None 
DFA_SAVE_HERE = "../experiments/boolean/dfa_object1.pkl"

BATCH_SIZE = 10
MODEL_FILE = "../experiments/boolean/models/boolean.h5"
TESTDATA_FILE = "../experiments/boolean/test_data_boolean.pkl" # TODO: add path to pickle file
K_NUM = 5
FRACTION = 0.001

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
		self.k_labels = list(range(self.k_num))
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

		## finding the sequence & char number in testing data corresponding 
		## to every state
		occur = self.find_occurance(kmeans.labels_)

		## cut down list naively  -- may replace with randomized sampling

		for state in occur:
			threshold = int(FRACTION * len(occur[state]))
			replacement = []
			for i in range(threshold):
				replacement.append(occur[state][i])
			occur[state] = replacement

		## Instantiate dfa object
		dfa = DFA()
		dfa.set_omega(self.alphabet)
		dfa.set_Q(self.k_labels)
		dfa.initiate_delta()

		# TODO: set q_0? Also not sure about F

		######################################################################
		## create state objects and visited/unvisited markers
		# states = [State("s{}".format(i)) for i in range(self.k_num)]
		# visited = [False]*self.k_num

		# ## initializing variables for BFS
		# current_state_num = 0
		# visited[0] = True
		# q = deque()
		# q.append(states[0])

		# # create DFA object
		# dfa = DFA()
		# dfa.set_omega(self.alphabet)
		# dfa.set_Q(states)
		# dfa.set_q_0(states[0])
		
		######################################################################

		prev_input = None

		total_processes = sum([len(element) for element in occur.values()])
		print("{} total processes to be run".format(total_processes))

		import ipdb; ipdb.set_trace()

		block = total_processes/50.0
		counter = 0
		
		for cluster in occur.keys():

			# occur[cluster] holds list of tuples
			instances = occur[cluster]

			for (seq_num, char_num) in instances:
				## Display loading bar:
				counter += 1
				print("#" * int(counter/block) + " " * (50 - int(counter/block))+ "| {} / {}".format(counter, total_processes))

				## Find the corresponding section of the test data
				chopped = self.data[seq_num][:char_num+1]
				prev_input = chopped[-1]

				print("prev character: {}".format(self.arr2char(prev_input.tolist())))

				for i in self.next_alphabet(prev_input):
					print("trying character: {}".format(self.arr2char(i.tolist())))
					

					new_seq = np.vstack((prev_input, i))

					if new_seq.shape[0] > 50:
						over = new_seq.shape[0] - 50

						## check if the first "over" elements are X's, if not,
						## this is an unsuable example
						if all([np.array_equal(new_seq[i], np.array([1,0,0,0,0])) for i in range(over)]):
							new_seq = new_seq[over:]

						else:
							print("".join([self.arr2char(row) for row in new_seq]) + " does not contain enough X's")
							pass # skip out of for loop


					elif new_seq.shape[0] < 50:
						over = 50 - new_seq.shape[0]
						insert = np.tile(np.array([1,0,0,0,0]), (over, 1))
						new_seq = np.vstack((insert, new_seq))


					## putting sequence into block of size BATCH_SIZE
					train_copy = self.data[seq_num:seq_num + BATCH_SIZE].copy()
					train_copy[0] = new_seq

					## Find hidden state
					new_states = ex.get_states(train_copy, 
							batch_size=BATCH_SIZE, unshuffle=True)
					next_s = new_states[0, -16:]

					## using SVM to predict the quantized state for hidden state
					next_state_num = kmeans.predict(next_s[np.newaxis,:])[0]

					## increment the (state1, transition, state2) in dfa in delta function
					transition = self.arr2char(i.tolist())
					dfa.add_delta(cluster, transition, next_state_num)

		# convert counts in dfa.delta to probabilities.
		dfa.delta_count2prob()

		return dfa

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

		if np.array_equal(prev_input, np.array([1,0,0,0,0])): # if "X"
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

	def find_occurance(self, labperstep):
		"""
		Helper function for extract.
		
		Inputs:
			labperstep (list): the 
		
		Returns:
			first_occur: a dictionary where the keys corresponds to the 
		"""

		occurrence = {state:[] for state in self.k_labels}

		for (idx, i) in enumerate(labperstep):
			# idx can be split into seq_num = idx//50 and char_num = idx%50
			seq_char = (idx//50, idx%50)
			occurrence[i].append(seq_char)

		assert len(occurrence) == self.k_num, \
			 "Length of dictionary doesn't match the number of clusters."

		return occurrence
	

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

		# variable used for calculating probabilites:
		self.count_s1t = {}

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
				self.count_s1t [(state1, transition)] = 0
				for state2 in self.Q:
					# initializing all values to None for now.
					self.delta [(state1, transition, state2)] = 0

	def add_delta(self, state1, transition, state2):
		# dictionary mapping (state1, transition) -> state2

		# delta should have all possible transition in the keys of the dictionary
		# e.g. (state1, transition, state2)

		if (state1, transition, state2) in self.delta:
			self.delta [(state1, transition, state2)] += 1
			self.count_s1t [(state1, transition)] += 1
		else:
			raise KeyError("{} hasn't been stored in the dictionary".format((state1, transition, state2)))

	def set_q_0(self, q_0):
		self.q_0 = q_0

	def add_F(self, F):
		# TODO
		if self.F == None:
			self.F = set([F])

		else:
			self.F.add(F)

	def delta_count2prob(self):
		for key in list(self.delta.keys()):
			if self.delta[key] == 0:
				del self.delta[key]
			else:
				self.delta[key] /= float(self.count_s1t[(key[0], key[1])])

		
	def __str__(self):
		"""
		Some way to print out the states
		"""
		return "Q: {} \n omega: {} \n delta: {} \n q_0: {} \n F: {}\n".\
			format(self.Q, self.omega, self.delta, self.q_0, self.F)


if __name__ == "__main__":
# def main():
	"""
	model_filename (string): path to the model
	"""

	if CACHED == None:
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

		dfa = fsm_ex.extract()

		pickle.dump(dfa, open(DFA_SAVE_HERE, "wb"))

	else:
		dfa = pickle.load(open(CACHED, "rb"))





# if __name__ == "__main__":

# 	main()

