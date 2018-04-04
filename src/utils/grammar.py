# Copyright: the Kevin

from collections import defaultdict
import random

import numpy as np

def weighted_choice(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

class CFG(object):
    def __init__(self):
        self.prod = defaultdict(list)

    def add_prod(self, lhs, rhs):
        """ Add production to the grammar. 'rhs' can
            be several productions separated by '|'.
            Each production is a sequence of symbols
            separated by whitespace.
            Usage:
                grammar.add_prod('NT', 'VP PP')
                grammar.add_prod('Digit', '1|2|3|4')
        """
        prods = rhs.split('|')
        for prod in prods:
            self.prod[lhs].append(tuple(prod.split()))

    def set_init_prod(self, prod):
        self.init_prod = prod
        # Dummy rule to reduce the stack size when parsing
        self.add_prod('DATASET', prod + ' | DATASET ' + prod)

    def gen_random_convergent(self, symbol=None,
            cfactor=0.25, pcount=defaultdict(int)):

        sentence = ''
        if symbol == None:
            symbol = self.init_prod

        # The possible productions of this symbol are weighted
        # by their appearance in the branch that has led to this
        # symbol in the derivation
        #
        weights = []
        for prod in self.prod[symbol]:
            if prod in pcount:
                weights.append(cfactor ** (pcount[prod]))
            else:
                weights.append(1.0)

        rand_prod = self.prod[symbol][weighted_choice(weights)]

        # pcount is a single object (created in the first call to
        # this method) that's being passed around into recursive
        # calls to count how many times productions have been
        # used.
        # Before recursive calls the count is updated, and after
        # the sentence for this call is ready, it is rolled-back
        # to avoid modifying the parent's pcount.
        #
        pcount[rand_prod] += 1

        for sym in rand_prod:
            # for non-terminals, recurse
            if sym in self.prod:
                sentence += self.gen_random_convergent(
                                    symbol=sym,
                                    cfactor=cfactor,
                                    pcount=pcount)
            else:
                sentence += sym + ' '

        # backtracking: clear the modification to pcount
        pcount[rand_prod] -= 1
        return sentence

    def parse_features(self, sequence):
        # Creates features for each rule
        features = {}
        for rule in self.prod:
            features[rule] = [0] * len(sequence)

        stack = []

        def rule_reduce(rule, stack):

            max_len_rule = max(len(r) for r in self.prod[rule])
            stack_top = max(0, (len(stack) - max_len_rule))

            # For each element in stack
            for i in range(stack_top, len(stack)):

                # For each production in the right hand side of the rule
                for expr in self.prod[rule]:

                    # Skips if production is too long
                    if len(expr) > len(stack) - i:
                        continue

                    # Tests match
                    match = all(expr[k] == stack[i+k][0]
                                    for k in range(len(expr)))

                    # If found
                    if match:

                        # computes the start and end of the sequence
                        start = stack[i][1]
                        end = stack[i+len(expr)-1][2]

                        # updates the features
                        for k in range(start, end + 1):
                            features[rule][k] += 1

                        # updates the stack
                        new_stack = stack[:i] + \
                                    [(rule, start, end)] + \
                                    stack[i+len(expr):]

                        return True, new_stack

            return False,stack


        # For each character:
        for i,s in enumerate(sequence):

            if i%1000 == 0:
                print 'Parsed', i, 'symbols out of', len(sequence)

            # Appends to stack
            start_ix = 0 if len(stack) == 0 else stack[-1][2] + 1
            end_ix = start_ix
            stack.append((s,start_ix,end_ix))

            # Reduces
            reduced = True
            while reduced:
                for rule in self.prod:
                    reduced,stack = rule_reduce(rule, stack)
                    if reduced:
                        break

        print 'Done parsing'
        return features


class GrammarUseCase:

    def __init__(self, prod_rules, init_prod):
        self.cfg = CFG()
        for r in prod_rules:
            self.cfg.add_prod(r[0], r[1])
        self.cfg.init_prod = init_prod

    def gen_sequence(self, n_sentences, cfactor=0.25):
        print 'Generating raw text size:', n_sentences
        output = ''
        for i in xrange(n_sentences):
            output += self.cfg.gen_random_convergent(cfactor = cfactor)
        out = [s for s in output.split(' ') if s != '']
        print('Generated sequence - %d characters' % len(out))
        return out

    def gen_depth_feat(self):
        def F(seq):
            dic_f = self.cfg.parse_features(seq)
            f = np.array(dic_f.values())
            f = f.transpose()
            feat_names = ['prod_depth_' + r[0] for r in self.cfg.prod]
            return f,feat_names
        return F

    def gen_bin_feat(self):
        # Note - repeats the feature function above!
        def F(seq):
            dic_f = self.cfg.parse_features(seq)
            f = np.array(dic_f.values()).transpose()
            f = f > 0
            f = f.astype(int)
            feat_names = ['prod_' + r for r in self.cfg.prod]
            return f,feat_names
        return F

    def gen_bin_feat_trig(self):
        # Note - repeats the feature function above!
        def F(seq):
            dic_f = self.cfg.parse_features(seq)
            f = np.array(dic_f.values()).transpose()

            f = f > 0
            f = f.astype(int)

            f[0] = f[1]
            f[1:] = abs(f[1:] - f[:-1])

            feat_names = ['trig_' + r for r in self.cfg.prod]
            return f,feat_names
        return F

    def gen_bin_depth_feat(self):
        pass