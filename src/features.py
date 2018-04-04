import numpy as np
import re
#################################################
# Generic function to extract a set of features #
#################################################
class FeatureFrame:

    def __init__(self, feature_functions, sequence):
        self.feature_functions = feature_functions
        self.sequence = sequence
        self._names = None
        self._values = None

    @property
    def names(self):
        if self._names is None:
            self.extract()
        return self._names

    @property
    def values(self):
        if self._values is None:
            self.extract()
        return self._values

    @property
    def data(self):
        if self._names is None or self._values is None:
            self.extract()
        return self._names, self._values

    def extract(self):
        # TO DO: speed up - np,concatenate is super slow
        print 'Generating feature scores'
        features = []
        colnames = []
        for i_F,F in enumerate(self.feature_functions):
            print 'Running feature ' + str(i_F) + ' out of ' + \
                     str(len(self.feature_functions))

            # Runs the actual function
            out = F(self.sequence)

            # Handles different types of output:
            if isinstance(out, np.ndarray):
                values = out
                if len(out.shape) == 1:
                    names = [F.__name__]
                elif len(out.shape) == 2:
                    names = [F.__name__ + str(i) for i in range(out.shape[1])]
                else:
                    raise ValueError('Incorrect feature function output')

            elif isinstance(out, tuple) and len(out) == 2:
                values,names = out
                if isinstance(names,basestring):
                    names = [names]

            else:
                raise ValueError('Incorrect feature function output')

            if len(values.shape) < 2:
                values = values[:,np.newaxis]

            colnames += names
            features.append(values)

            pnames = names if len(names) < 5 else names[:5] + ["..."]
            print 'Added features', pnames

        print 'Tidying...'
        out = np.concatenate(features, axis = 1)

        print 'Computed feature matrix, with shape:', out.shape
        print 'Snippet of the features'
        print ''.join(self.sequence)[-50:]
        print colnames
        print out[-50:,:]

        self._names = colnames
        self._values = out[...]


###############################
# Different feature functions #
###############################
def alphabet():
    def F(strings):
        chars = sorted(set(strings))
        char2int = {c:i for i,c in enumerate(chars)}

        features = np.zeros((len(strings), len(char2int)))
        for i,c in enumerate(strings):
            features[i,char2int[c]] = 1

        names = ['char_' + str(c) for c in char2int]
        return features, names
    return F

def regex(expr):
    def F(strings):
        if type(strings) == list:
            strings = ''.join(strings)

        out = np.zeros((len(strings)))
        p = re.compile(expr)
        for m in p.finditer(strings):
            s = m.start()
            L = len(m.group())
            out[s:s+L] = 1
        return out, 'expr' + repr(expr)
    return F

def alphanum():
    def F(strings):
        out = []
        for i,c in enumerate(strings):
            out.append(c.isalnum())
        out = np.array(out).astype(int)
        return out, 'is_alphanum'
    return F

def word_start_end():
    def F(strings):
        word = []
        for i,c in enumerate(strings):
            word.append(c.isalnum())
        word = np.array(word).astype(int)
        s = np.concatenate([[0], word])
        start = s[1:] > s[:-1]
        e = np.concatenate([word, [0]])
        end =  e[:-1] > e[1:]
        out = np.logical_or(start, end).astype(int)
        return out, 'word_lim'
    return F

def is_enclosed(s1, s2):
    def F(strings):
        indentation = np.zeros((len(strings)))
        cur_indent = 0
        for i in range(len(strings)):
            if strings[i] == s1:
                cur_indent += 1
            if strings[i] == s2 and cur_indent > 0 :
                cur_indent -= 1
            indentation[i] = cur_indent
        return indentation, 'dep_' + repr(s1) + repr(s2)
    return F

def is_enclosed_bin(s1, s2):
    def F(strings):
        indentation = np.zeros((len(strings)))
        cur_indent = 0
        for i in range(len(strings)):
            if strings[i] == s1:
                cur_indent += 1
            if strings[i] == s2 and cur_indent > 0 :
                cur_indent -= 1
            indentation[i] = cur_indent

        num_indent = int(np.max(indentation))
        bin_indent = np.zeros((indentation.shape[0], num_indent))
        for i in range(indentation.shape[0]):
            indent = int(indentation[i])
            bin_indent[i,:indent]=1

        names = ['dep_'+repr(s1)+repr(s2)+'_'+str(i) \
                    for i in range(1,num_indent + 1)]

        return bin_indent, names
    return F

def is_enclosed_str_nodepth(s1, s2):
    def F(strings):
        if type(strings) == list:
            strings = ''.join(strings)
        out = np.zeros((len(strings)))
        is_enclosed = False
        for i in range(len(strings)):
            if i > len(strings) - len(s1):
                break
            elif strings[i:i+len(s1)] == s1 and not is_enclosed:
                is_enclosed = True
            elif i > len(strings) - len(s2):
                break
            elif strings[i:i+len(s2)] == s2 and is_enclosed:
                is_enclosed = False
            out[i] = is_enclosed
        return out, 'enc_' + repr(s1) + repr(s2)
    return F

def line_indent_level():
    def F(strings):
        indent = 0
        indentation = []
        for char in strings:
            if char == '\n':
                indent = 0
            if char == '\t':
                indent += 1
            indentation.append(indent)
        indentation = np.array(indentation)

        num_indent = int(np.max(indentation))
        bin_indent = np.zeros((indentation.shape[0], num_indent))
        for i in range(indentation.shape[0]):
            indent = int(indentation[i])
            bin_indent[i,:indent+1]=1

        names = ['indent'+str(i) for i in range(num_indent)]

        return bin_indent, names
    return F

def line_char_pos():
    def F(strings):
        positions = np.zeros((len(strings)))
        cur_char = 0
        for i,c in enumerate(strings):
            positions[i] = cur_char
            if c == '\n':
                cur_char = 0
            else:
                cur_char += 1
        return positions, 'line_pos'
    return F

def line_char_pos_bin(step_size=10, pmax=50):
    def F(strings):
        # Gets positions
        positions = np.zeros((len(strings)))
        cur_char = 0
        for i,c in enumerate(strings):
            positions[i] = cur_char
            if c == '\n':
                cur_char = 0
            else:
                cur_char += 1

        # Encodess
        positions[positions > pmax] = pmax
        positions = positions // step_size
        positions = positions.astype(int)
        n_bins = np.max(positions) + 1
        binpos = np.zeros((positions.shape[0], n_bins))
        for i in range(positions.shape[0]):
            binpos[i,positions[i]]=1
        names = ['pos'+ str(i * step_size) + '_' +  str((i+1) * step_size) \
                    for i in range(n_bins)]
        return binpos, names
    return F

def n_grams(n, max_hash, delay=0):
    def F(strings):
        features = np.zeros((len(strings), max_hash))
        for i in range(len(strings)):
            f_start = i - delay
            f_end   = f_start + n
            if not 0 <= f_start < len(strings) or \
               not 0 <= f_end < len(strings):
                continue
            str_list = strings[f_start:f_end]
            h = hash(tuple(str_list)) % max_hash
            features[i,h]=1
        names = ['n_gram_'+str(h)+'_lag_'+str(delay)
            for h in range(max_hash)]
        return features,names
    return F

def char_word_hash(max_hash):
    def F(chars):
        features = np.zeros((len(chars), max_hash))
        start_pos = 0
        cur_word = ''
        for i,c in enumerate(chars):
            if c.isalnum():
                if start_pos is -1:
                    start_pos = i
                cur_word += c
            else:
                h = hash(cur_word) % max_hash
                features[start_pos:start_pos+len(cur_word),h]=1
                start_pos = -1
                cur_word = ''
        h = hash(cur_word) % max_hash
        features[start_pos:start_pos+len(cur_word),h]=1

        names = ['n_gram_'+str(h) for h in range(max_hash)]
        return features,names
    return F

def sentence_ends(char, ignored_char):
    def F(strings):
        f = np.zeros(len(strings))
        found = False
        for i in range(len(strings))[::-1]:
            cur_str = strings[i]
            is_char = cur_str == char
            is_word = cur_str.isalnum()
            is_ignored = cur_str in ignored_char or len(cur_str) > 1
            is_separator = not is_word and not is_char and not is_ignored
            found = is_char or (found and not is_separator)
            if found:
                f[i]=1
        names = ['ends' + char]
        return f,names
    return F

def presuffix(nchar, prefix=True):
    def F(words):
        # Produces values
        presuffixes = []
        for w in words:
            n = min(nchar,len(w))
            if prefix:
                presuffix = w[:n]
            else:
                presuffix = w[-n:]
            presuffixes.append(presuffix)

        all_presuffixes = list(set(presuffixes))
        presuffix_code = {p:i for i,p in enumerate(all_presuffixes)}

        features = np.zeros((len(strings), len(all_presuffixes)))
        for i,p in enumerate(all_presuffixes):
            features[i, presuffix_code[p]] = 1

        # Creates labels
        s = 'pre' if prefix else 'suf'
        names = [s+'_'+p for p in all_presuffixes]
        return features,names
    return F

def bool_fsm_states(char2int):
    def F(seq):
        n_states = 7
        n_symbols = len(char2int)

        print len(seq) 
        features = np.zeros((len(seq), n_states))
        # trans_tbl = np.zeros(n_states, n_symbols)
        trans_tbl = [[0, 2, 1, -1, 1],
                    [0, -1, -1, 4, 3],
                    [0, -1, -1, 6, 5],
                    [0, 2, 1, -1, -1],
                    [0, 1, 1, -1, -1],
                    [0, 2, 2, -1, -1],
                    [0, 2, 1, -1, -1]]

        curr_state = 0
        for i, char in enumerate(seq):
            curr_state = trans_tbl[curr_state][char2int[char]]
            features[i][curr_state] = 1

        names = ['S' + str(i) for i in range(n_states)]
        return features, names
    return F

            

        

