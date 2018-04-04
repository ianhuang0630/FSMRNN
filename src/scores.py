from sklearn import svm, linear_model
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn import svm

from sklearn.metrics import normalized_mutual_info_score

from scipy import stats

import numpy as np


class Score:

    def preprocess(self, states):
        return states

    def score_layer():
        pass

    def check(self, feature, states):
        pass

    def compute(self, feature, states):
        pass

## Single column scores
class SingleScore(Score):

    def score_cell(self, feat_values, states):
        n_neurons = states.shape[1]
        out = np.empty((n_neurons))
        for i_neuron in range(n_neurons):
            out[i_neuron] = self.run(feat_values,states[:,i_neuron])
        return out

    def run(self, feature, states):
        self.check(feature, states)
        states = self.preprocess(states)
        try:
            score = self.compute(feature, states)
        except ValueError as e:
            print e
            return None
        return score

    def check(self, feature, states):
        if feature.shape[0] != states.shape[0]:
            msg = 'Feature has ' + str(feature.shape[0]) + ' rows, while ' + \
                    'states has ' + str(states.shape[0]) + ' rows'
            raise ValueError(msg)

        if len(feature.shape) > 1:
            raise ValueError('Feature vector has ' + len(feature.shape) + \
                             ' dimensions instead of 1')

        if len(states.shape) > 1:
            raise ValueError('State vector has ' + len(states.shape) + \
                             ' dimensions instead of 1')


class Correlation(SingleScore):

    def compute(self, feature, states):
        is_nan  = np.isnan(states) | np.isnan(feature)
        states  = states[~is_nan]
        feature = feature[~is_nan]
        if np.unique(states).size < 2 or np.unique(feature).size < 2:
            return 0
        r = abs(np.corrcoef(states, feature)[1,0])
        return r

class SingleLogReg(SingleScore):

    def compute(self, feature, states):
        X = states[:,np.newaxis]
        y = feature

        # Missing values management
        x_na_rows = np.apply_along_axis(lambda x: np.isnan(x).any(), 1, X)
        y_na_rows = np.isnan(y)
        na_rows = x_na_rows | y_na_rows

        X = X[~na_rows,:]
        y = y[~na_rows]

        # Model training
        try:
            # TO DO: switch to L0
            reg = LogisticRegression()
            #reg = LogisticRegression(class_weight='balanced')
            reg.fit(X,y)
            f1 = cross_val_score(reg, X, y, scoring='f1').mean()
#            neuron_scores = np.absolute(reg.coef_[0,:])
            model_score = f1
        except ValueError:
            print 'Cant compute score'
            neuron_scores = np.repeat(0, X.shape[1])
            model_score = 0

        return model_score

class SVM(SingleScore):

    def compute(self, feature, states):
        X = states[:,np.newaxis]
        y = feature

        # Missing values management
        x_na_rows = np.apply_along_axis(lambda x: np.isnan(x).any(), 1, X)
        y_na_rows = np.isnan(y)
        na_rows = x_na_rows | y_na_rows

        X = X[~na_rows,:]
        y = y[~na_rows]

        # Model training
        try:
            reg = svm.SVC()
            reg.fit(X,y)
            f1 = cross_val_score(reg, X, y, scoring='f1').mean()
            model_score = f1
        except ValueError:
            print 'Cant compute score'
            neuron_scores = np.repeat(0, X.shape[1])
            model_score = 0

        return model_score

class MutualInformation(SingleScore):

    def __init__(self, threshold=.5):
        self.threshold=threshold

    def preprocess(self, states):
        states = np.absolute(states)
        states = states > self.threshold
        return states

    def compute(self, feature, states):
        is_nan  = np.isnan(states) | np.isnan(feature)
        states  = states[~is_nan]
        feature = feature[~is_nan]
        s = normalized_mutual_info_score(states,feature)
        return s



# To do: SVM
# To do: booleans
# To do: Phi!

## Multi-scores
class MultiScore(Score):

    def score_cell(self, feat_values, states):
        return self.run(feat_values,states)

    def run(self, feature, states):
        self.check(feature, states)
        states = self.preprocess(states)
        try:
            score = self.compute(feature, states)
        except ValueError as e:
            print e
            return None
        return score

    def check(self, feature, states):
        if feature.shape[0] != states.shape[0]:
            msg = 'Feature has ' + str(feature.shape[0]) + ' rows, while ' + \
                    'states has ' + str(states.shape[0]) + ' rows'
            raise ValueError(msg)

        if len(feature.shape) > 1:
            raise ValueError('Feature vector has ' + len(feature.shape) + \
                             ' dimensions instead of 1')

        if len(states.shape) != 2:
            raise ValueError('State vector has ' + len(states.shape) + \
                             ' dimensions instead of 2')

class MultiLogReg(MultiScore):

    def __init__(self, f1_threshold = .75):
        self.f1_threshold = f1_threshold

    def compute(self, feature, states):
        X = states
        y = feature

        # # Missing values management
        # x_na_rows = np.apply_along_axis(lambda x: np.isnan(x).any(), 1, X)
        # y_na_rows = np.isnan(y)
        # na_rows = x_na_rows | y_na_rows

        # X = X[~na_rows,:]
        # y = y[~na_rows]

        # Model training
        # print('Fitting model')
        try:
            # TO DO: switch to L0
            #reg = LogisticRegressionCV(Cs=5, penalty='l2', class_weight='balanced')
            reg = LogisticRegressionCV(Cs=5, penalty='l1', solver='saga', class_weight='balanced')
            reg.fit(X,y)
            f1 = cross_val_score(reg, X, y, scoring='f1').mean()
            print 'F1 score:', f1
            if f1 >= self.f1_threshold:
                neuron_scores = np.absolute(reg.coef_[0,:])
                neuron_scores = stats.zscore(neuron_scores)
            else:
                neuron_scores = np.zeros_like(reg.coef_[0,:])

        except ValueError as err:
            print 'ERROR Cant compute score:', err
            neuron_scores = np.repeat(0, X.shape[1])
            model_score = 0

        return neuron_scores

class MultiLogRegL2(MultiScore):

    def __init__(self, f1_threshold = .75):
        self.f1_threshold = f1_threshold

    def compute(self, feature, states):
        X = states
        y = feature

        try:
            reg = LogisticRegressionCV(Cs=5, penalty='l2', class_weight='balanced')
            reg.fit(X,y)
            f1 = cross_val_score(reg, X, y, scoring='f1').mean()
            print 'F1 score:', f1
            if f1 >= self.f1_threshold:
                neuron_scores = np.absolute(reg.coef_[0,:])
                neuron_scores = stats.zscore(neuron_scores)
            else:
                neuron_scores = np.zeros_like(reg.coef_[0,:])

        except ValueError as err:
            print 'ERROR Cant compute score:', err
            neuron_scores = np.repeat(0, X.shape[1])
            model_score = 0

        return neuron_scores

class MultiMutualInfoClass(MultiScore):

    def __init__(self, discrete_features=False):
        self.discrete = discrete_features

    def compute(self, feature, states):
        X = states
        y = feature
        out =  mutual_info_classif(X,y,discrete_features=self.discrete)
        return out

class MultiFTestClass(MultiScore):

    def compute(self, feature, states):
        X = states
        y = feature
        out =  f_classif(X,y)[0]
        return out


# Unit test scores #
## Multi-scores
class TestScore:

    def preprocess(self, states):
        return states

    def score(self, feature, states):
        self.check(feature, states)
        states = self.preprocess(states)
        try:
            score = self.compute(feature, states)
        except ValueError as e:
            print e
            return None
        return score

    def check(self, feature, states):
        if feature.shape[0] != states.shape[0]:
            msg = 'Feature has ' + str(feature.shape[0]) + ' rows, while ' + \
                    'states has ' + str(states.shape[0]) + ' rows'
            raise ValueError(msg)

        if len(feature.shape) > 1:
            raise ValueError('Feature vector has ' + str(len(feature.shape)) + \
                             ' dimensions instead of 1')

        if len(states.shape) != 2:
            raise ValueError('State vector has ' + str(len(states.shape)) + \
                             ' dimensions instead of 2')

class LogRegF1(TestScore):

    def compute(self, feature, states):
        X = states
        y = feature

        try:
            #reg = LogisticRegressionCV(Cs=5, penalty='l2', class_weight='balanced')
            reg = LogisticRegressionCV(class_weight='balanced')
            reg.fit(X,y)
            f1 = cross_val_score(reg, X, y, scoring='f1').mean()
            return f1

        except ValueError as err:
            print 'ERROR Cant compute score:', err
            return 0