import numpy as np

def encode(sequence):
    print 'Encoding'
    all_characters = sorted(set(sequence))
    char2int =  {c:i for i, c in enumerate(all_characters)}
    int2char =  {i:c for i, c in enumerate(all_characters)}
    print "Total vocabulary len_sequence: ", len(all_characters)
    encoded_seq = [char2int[x] for x in sequence]
    return encoded_seq, char2int, int2char

def encode_split(sequence1, sequence2):
    print 'Encoding'
    all_characters = sorted(set(sequence1 + sequence2))
    char2int =  {c:i for i, c in enumerate(all_characters)}
    int2char =  {i:c for i, c in enumerate(all_characters)}
    print "Total vocabulary size: ", len(all_characters)
    encoded_seq1 = [char2int[x] for x in sequence1]
    encoded_seq2 = [char2int[x] for x in sequence2]
    return encoded_seq1, encoded_seq2, char2int, int2char

def vectorize(seq, window_size):
    out = []
    for i in range(len(seq)-window_size+1):
        out.append(seq[i:i+window_size])
    out = np.array(out)
    return out

def one_hot_encode_seq(seq, n_chars = None):
    if n_chars is None:
        n_chars = max(seq)+1
    X = np.zeros((len(seq),n_chars), dtype=np.bool)
    for i, char in enumerate(seq):
        X[i, char] = 1
    return X

def one_hot_encode_matrix(mat):
    # One hot encoding for the X
    X = np.zeros((mat.shape[0], mat.shape[1], np.amax(mat)+1), dtype=np.bool)
    for i,seq in enumerate(mat):
        for c,char in enumerate(seq):
            X[i,c,char] = 1
    return X

def shuffle_indices(len_sequence, len_batch):
    len_sequence = len_sequence - len_sequence%len_batch

    shuffle_i = [None] * len_sequence
    offset = 0
    pos    = 0
    for n in range(len_sequence):
        if pos + offset >= len_sequence:
            pos = 0
            offset += 1
        shuffle_i[pos + offset] = n
        pos += len_batch

    return shuffle_i

def unshuffle_indices(len_sequence, len_batch):
    s = shuffle_indices(len_sequence, len_batch)
    return np.argsort(s)