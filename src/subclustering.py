import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from FSMextractor import SampleBasedTransExtractor
from tqdm import tqdm
from multiprocessing import Pool

def default_metric(states, labels, random_seed=None):
    """ Returning the silhouette coefficient.
    """
    
    if states.shape[0] > 1000:
        chooseidx=np.random.choice(states.shape[0], 1000)
        a = states[chooseidx]
        b = labels[chooseidx]

    else:
        a = states
        b = labels

    return silhouette_score(a,b)


def f (matrix, rng, delt, col=0):
    """ Helper function to entropy_based().
    Inputs:
        matrix(np.array):
        rng (tuple):
        delt (float):
        col (int): column number

    returns:
        counts (list):

    """
    if not matrix.shape[0]:
        return [0]

    if col == matrix.shape[1]-1: 
        # if references the last column, then recursion nolonger needed
        
        start, end = rng[0], delt
        counts = []
        while start < rng[1]:
            cnt=0
            for row in matrix:
                if row[col] >= start and row[col] < end:
                    cnt += 1
            counts.append(cnt)
            start += delt
            end += delt
        return counts

    else:

        (start,end) = (rng[0],delt)
        counts = []

        while start < rng[1]:
            # concat the matrix to only include the relevant rows
            mat2 = []

            for row in matrix:
                if row[col] >= start and row[col] < end:
                    # append it to mat2
                    mat2.append(row)
            mat2 = np.array(mat2)

            # call f on concatted matrix, same rng, delt, but col += 1.
            cnt = f(mat2, rng, delt, col=col+1)
            counts.extend(cnt)

            # incrementing start and end
            start += delt
            end += delt

        return counts
        

def entropy_based(states, rng=(-1.0, 1.0), delt=0.1, random_seed=None):
    """ Entropy_based metric, based on:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.1465&rep=rep1&type=pdf
    """
    
    if random_seed!=None:
        np.random.seed(random_seed)

    # may have to randomize sampling of states if it's too big. 
    if states.shape[0] > 1000:
        chooseidx=np.random.choice(states.shape[0], 1000)
        sampled_states = states[chooseidx]

    else:
        sampled_states = states

    counts = f(sampled_states, rng, delt)
    
    entropy = -sum([x/float(sampled_states.shape[0]) * 
                    np.log(x/float(sampled_states.shape[0])) 
                    for x in counts if x > 0])
    
    return entropy

def kmeans_wrapper(aoi):
    km = KMeans(5)
    km.fit(aoi)
    return km.labels_
    

ACT_MAT=None

def multiprocessing_f(params):
    """ Function used to parallelize the clustering_funct
    Input:
        params (tuple): first item is a list of columns and the second is delt 
            param in the entropy_based subspace function. 

    Returns:
        entropy (float): the entropy value
    """

    print("multirpocess function {}".format(params[0]))
    column_ind = params[0]
    delt = params[1]

    assert len(column_ind) > 0, "column_ind should have positive length"
    assert type(ACT_MAT) == np.ndarray, "{}".format(type(ACT_MAT))

    # calculating entropy
    aoi = ACT_MAT [:, np.array(column_ind)]
    ent_meas = entropy_based(aoi, delt=delt)
    entropy = (column_ind, ent_meas)
    return entropy

def bruteforce_prunesubspace(hid_states, delt=0.1, parallel=False,
                            num_process=None):

    """ A bruteforce approach that uses entropy to prune the subspace before 
        clustering.
    
    Inputs:
        hid_states(np.array): matrix holding neuron activations
        metric (func): function used 
        delt (float, optional): Size of the step used in entropy evaluation
        parallel (float, optional): True if multiprocessing is to be used,
            False otherwise.
        num_process (int, optional): if parallel is true, then num_process is 
            the number of cores that the multiprocessing will be using. 
    returns:
        subspace_clusters (set):

    """
    global ACT_MAT
    subspace_clusters = set([])
    
    num_neurons = hid_states.shape[1]
    column_indices = range(num_neurons)

    if parallel:
        # initialize the global activation matrix
        ACT_MAT = hid_states
        # generate tuple 
        neuron_combos = []
        print("Calculating combinations")
        
        # TODO: use multiprocessing in generating neuron combos too?
        for i in range(1, num_neurons + 1):
            for j in combinations(column_indices, i):
                # j is a tuple
                assert type(j) == tuple
                neuron_combos.append(j)

        delt_list = [delt] * len(neuron_combos)
    
        print("Starting multiprocessing:")
        p = Pool(processes = num_process)
        # p = ProcessingPool()

        params = zip(neuron_combos, delt_list)
        # entropy_lst = p.imap_unordered(multiprocessing_f, params)
        
        # using map instead:
        entropy_lst = p.map(multiprocessing_f, params)
        
        p.close()
        print (entropy_lst)
        subspace_clusters = set(entropy_lst)
        
    else: # in the case of computation on a single cpu

        for i in range(1, num_neurons+1):
            print("Considering collection of {} neurons".format(i))
            for j in tqdm( combinations(column_indices, i)):
                comb = np.array(j)
                aoi = hid_states[:, comb]
                  
                # finding the entropy measurement 
                ent_meas = entropy_based(aoi, delt=delt)
                subspace_clusters.add((j, ent_meas))
    
    return subspace_clusters

def bruteforce_extract_transitions(model, data, subspace_clusters):
    """
    Inputs:
        model (keras.Model): model of interest
        data (np.array): data being sampled
        subspace_clusters (dict): values hold column numbers of data in 
            subspace, values hold the predicted label of every timestep of 
            every sequence.
    """
    extractor = SampleBasedTransExtractor()
    transitions = {}
    for j in tqdm(subspace_clusters):
        aoi = data[:, np.array(j)]
        transitions[j]= extractor.extract_transition(model, aoi, 
                                                    subspace_clusters[j][0])
    
    return transitions


def bruteforce_clusterfirst(hid_states, metric, cluster_func, threshold):
    """ A bruteforce bottom up way to do subspace clustering, without any 
        pruning algorithms
    Input:
        hid_states (np.array): numpy array with N columns, where N = number of 
            neurons.
        metric (func): something to help evaluate the quality of clustering.
        cluster_func (func): clustering function which takes in a matrix.
            (note: not read as cluster f*ck.)
        threshold (float): function will continue to search for subspaces to 
            cluster that yields a metric that is less this threshold.
    Output:
        subspace_clusters (dic): key is a tuple of column numbers involved in 
            clustering, and value is a list contains which cluster each belongs 
            to.
        metric_per_subspace (dic): key is tupel of column numbers involved in 
            clustering, and value is the metric of clustering in that subspace.
    """
    subspace_clusters = {}
    metric_per_subspace = {}

    num_neurons = hid_states.shape[1]
    column_indices = range(num_neurons)
    known_failures = set([])
    
    # TODO: adding to subspace_clusters
    for i in range(1, num_neurons+1):
        for j in combinations(column_indices, i):
            # extracts part of matrix that will be clustered
            comb = np.array(j)
            
            # brutal pruning
            if set(comb[:-1]) not in known_failures:

                aoi = hid_states[:, np.array(j)]
    
                # TODO: change 5
                kmeans = KMeans(5)
                kmeans.fit(aoi)
                
                score = metric(aoi, kmeans.labels_)
                # Running clustering through the metric function
                if score > threshold:
                    # then add it to the dictioanry
                    print("Adding {} to potential subspace with metric of {}."
                            .format(j, score))
                    subspace_clusters[j] = kmeans.labels_
                    metric_per_subspace[j] = score 
                else:
                    print("neurons {} do not form a qualifying cluster, \
                            with metric {}".format(j, score)) 
                    known_failures.add(j)

    return subpace_clusters


if __name__ == "__main__":

    import sys
    sys.path.append('../src')
    
    import numpy as np 
    import pickle
    
    from FSMextractor import *
    
    from sklearn.metrics import silhouette_score
    
    BATCH_SIZE = 10
    MODEL_FILE = "../experiments/boolean/models/boolean.h5"
    TESTDATA_FILE = "../experiments/boolean/test_data_boolean.pkl" 
    
    model = load_model(MODEL_FILE)
    test_data = pickle.load(open(TESTDATA_FILE))
    
    
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
    
    # kmeans discretizer
    kMeans = kMeansQuantizer(5)
    
    # getting the sample based transition extractor
    sbtrans = SampleBasedTransExtractor(conv_func=arr2char)
    
    experiment = FSMExtractor(model, test_data, quantization=kMeans, transition=sbtrans, minimization=None)
    
    # get hidden states 
    states = experiment.get_hidden_states()

    states = states[:10, :4]
    print(bruteforce_prunesubspace(states, parallel=True, num_process=7))


