import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pickle

FILENAME_FEATURES = "features.pkl"
FILENAME_TEST_SEQ = "X_sequence.pkl"
FILENAME_STATES = "states.pkl"
TIME_START = 0
TIME_END = 600

VERBOSE = True

def apply_TSNE(f_states, v):
    """
    Inputs:
        f (string): name of file.
    Returns:
        embedded_states (np.array): an array of the embedded space of states.
    """
    states = pickle.load(open(f_states))
    # using TSNE to do embeddings
    embedded_states = TSNE(n_components=2, verbose=5*v).fit_transform(states[TIME_START:TIME_END,:])
    return embedded_states


def graph(es, test_set_crop):
    """
    Input:
        es (np.array): Array holding the embedded space of states
        ef (np.array): Array holding the embedded space of features

    Graphs out the resulting distribution of points
    """

    semantics_labels = semantics(test_set_crop)
    nl = nesting_level(test_set_crop)

    plt.figure()

    plt.subplot(2,1,1)
    plt.title("states grouped by semantics")

    # red = number, green = open brac, blue = close brac
    label = [0, 1, 2]
    colors = ['red', 'green', 'blue']
    semantics_label = ['number', 'open brackets', 'close brackets']
    plt.scatter(es[:,0], es[:,1], s=10, c=semantics_labels, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(semantics_label)

    plt.subplot(2,1,2)
    plt.title("states grouped by nesting level")

    colors = [int(i %23) for i in nl]

    plt.scatter(es[:,0], es[:,1], s=10, c=colors)

    cb = plt.colorbar()
    loc = np.arange(0, max(nl), max(nl)/float(len(set(colors))))
    cb.set_ticks(loc)
    cb.set_ticklabels(list(set(nl)))

    plt.show()

def nesting_level(test):
    """
    Input:
        test(list): list of input sequences
    Returns:
        levels (list): list of input sequences
    """

    levels = []
    counter = 0

    for element in test:
        if element == "(":
            counter += 1
            levels.append(counter)

        elif element == ")":
            levels.append(counter)
            counter -= 1

        else:
            levels.append(counter)

    return levels

def semantics(test):
    """
    Input:
        test (list): List of input sequences
    Returns:
        labels (list): Contains 0 if a number, 1 if open bracket, 2 if close bracket
    """
    labels = [None]*len(test)

    for (idx, element) in enumerate(test):
        if element == "(":
            labels[idx] = 1
        elif element == ")":
            labels[idx] = 2
        else: # if it is a number
            labels[idx] = 0

    assert all([element in {0,1,2} for element in labels]), "labels messed up"

    return labels

def main():

    es= apply_TSNE(FILENAME_STATES, VERBOSE)
    test_seq = pickle.load(open("X_sequence.pkl"))
    test_seq_crop = test_seq[TIME_START: TIME_END]

    # graphing according to the semantics of the input data
    graph(es, test_seq_crop)

    # grouping according to semantics of input


if __name__ == "__main__":
    main()
