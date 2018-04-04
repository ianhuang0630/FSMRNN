import numpy as np 
import pickle
from FSMextractor import *

from sklearn.metrics import silhouette_score


BATCH_SIZE = 10
MODEL_FILE = "../experiments/boolean/models/boolean.h5"
TESTDATA_FILE = "../experiments/boolean/test_data_boolean.pkl" 


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
        raise valueerror("sorry, can't recognize label")
    
test_data = pickle.load(open(TESTDATA_FILE))
     
# kmeans discretizer
kMeans = kMeansQuantizer(5)

# getting the sample based transition extractor
sbtrans = SampleBasedTransExtractor(conv_func=arr2char)

experiment = FSMExtractor(model, test_data, quantization=kMeans, transition=sbtrans, minimization=None)




# get hidden states 
states = experiment.get_hidden_states()


# add N random dimensions at the end

num_symbols = states.shape[0]
artificial_neurons = np.random.uniform(low=-1.0, high =1.0, size=(num_symbols,1))
states_alt = np.hstack((states, artificial_neurons))


# iterate between clustering and dimension elimination
# eliminating the dimensions with the highest variance from the centroids


fsm = {}
# XXX: number of neurons is default number of neurons (16) plus 1.
neuron_nums = np.arange(17)
silhouettes = []

for i in range(16): # XXX: delete neurons until only 1 is left.
    
    # printing separator
    print("ITERATION {}".format(i))


    # labels = experiment.quant_func.quantize(states_alt)
    
    # generating and saving fsm's
    
    fsm["iteration #{}".format(i)] = experiment.cluster_transition(states_alt)
    
    # extracting macrostate predictions
    labels = experiment.quant_func.getLabels()   
    centroids = experiment.quant_func.getCentroids() 


    # find out how well the hidden states are being clustered into macrostates
    silh = experiment.quant_func.getSilhouetteCoef(states_alt)
    silhouettes.append(silh)    
    print("Sillhouette coefficient: {}".format(silh))

    
    # finding a dimension to eliminate: the dimension that has the greatest average distance from
    # centroids of clustering
    # argmax_i \sum{j to k} 1/number_in_cluster * \sum{all i} x^(j)_i - C(j)(i) 
    
    delta = states_alt - centroids[labels]
    mean_delta = np.mean(delta, axis=0)

    jettison = np.argmax(mean_delta)
    print("Throwing out neuron number {}".format(neuron_nums[jettison]))
    
    # throw out activations for that neuron in states_alt
    neuron_nums = np.delete(neuron_nums, jettison)     
        
    states_alt = np.delete(states_alt, jettison, axis=1)
     


# TODO: plot sillhouettes

# TODO: generate histograms out of fsm's




