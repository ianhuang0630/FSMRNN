
# coding: utf-8

# # Libraries, Headers and Stuff

# In[1]:


import sys
sys.path.append('../../src')


# In[2]:


CACHED = True
CONTINUE = True

N_STATES = 256
N_EPOCHS =  50
BATCH_SIZE = 32

MAX_VOCAB_SIZE = -1
MAX_SEQUENCE_SIZE = -1

LAYERS_TO_CHECK = [0]

import scores
METRIC = scores.Correlation()

SAMPLES_VERIFY = 8
TEST_RATIO = .1

MODEL_NAME = 'thibz_layers_1_hidden_units_256_dropout_0.5'


# In[3]:


import datetime
import os

FOLDER_OUT = MODEL_NAME + '-' + str(datetime.datetime.today())[:16]
FOLDER_OUT = FOLDER_OUT.replace(' ','_')
if not os.path.exists(FOLDER_OUT):
    os.makedirs(FOLDER_OUT)

print FOLDER_OUT


# In[4]:


import numpy as np
import pandas as pd
import keras


# In[5]:


get_ipython().magic(u'load_ext rpy2.ipython')


# In[6]:


import random
random.seed(55555)


# # Creates dataset

# In[7]:


import kernelhelpers
reload(kernelhelpers)

import pickle

num_layer = 1

path = 'corpus/linux_kernel_val.txt'
model_weights = 'models/corpus_linux_kernel_val_architecture_stateful_type_char_layers_1_hidden_units_128_epoch_49_weights.h5'
model_settings_name = 'settings/corpus_linux_kernel_val_architecture_stateful_type_char_layers_1_hidden_units_128_settings.pickle'
new_batch_size = BATCH_SIZE
new_size = 1
new_time_skip = 1
is_stateful = True

# Loads the model
old_model,char2int,int2char,model_type =     kernelhelpers.load_setup(num_layer,
                path,
                model_weights,
                model_settings_name ,
                new_batch_size,
                new_size,
                new_time_skip,
                is_stateful)
    
raw_data = kernelhelpers.get_corpus('corpus/linux_input.txt')

dict_location = 'dicts/' + MODEL_NAME
with open(dict_location + '_char2int.pickle' , 'wb') as f:
    pickle.dump(char2int, f)
with open(dict_location + '_int2char.pickle' , 'wb') as f:
    pickle.dump(int2char, f)

print 'Number of charaters', len(raw_data)
print 'Number of charaters', len(set(raw_data))


# In[8]:


import utils.preprocess as pre

# Generates data
full_size = len(raw_data)
train_size = int((1-TEST_RATIO) * full_size)

train_sequence = raw_data[:train_size]
test_sequence = raw_data[train_size:]

raw_train_sequence = [char2int[c] for c in train_sequence]
raw_test_sequence  = [char2int[c] for c in test_sequence]

# If necessary, truncates:
if MAX_SEQUENCE_SIZE > 0:
    train_size = int(MAX_SEQUENCE_SIZE * (1-TEST_RATIO))
    test_size = int(MAX_SEQUENCE_SIZE * TEST_RATIO)
    raw_train_sequence = raw_train_sequence[:train_size]
    raw_test_sequence  = raw_test_sequence[:test_size]

print 'Training smybols:', len(raw_train_sequence)
print 'Testing smybols:', len(raw_test_sequence)
print 'Size vocabulay:', len(char2int)


# In[9]:


import numpy as np
import utils.preprocess as pre
reload(pre)

# Preprocesses training
# SPLITS
X_train_raw = raw_train_sequence[:-1]
y_train_raw = raw_train_sequence[1:]
# ENCODES
X_train = pre.one_hot_encode_seq(X_train_raw, n_chars=len(int2char))
y_train = pre.one_hot_encode_seq(y_train_raw, n_chars=len(int2char))
# SHUFFLES
train_indices = pre.shuffle_indices(len(X_train_raw), BATCH_SIZE)
y_train = y_train[train_indices]
# EXPANDS
X_train = X_train[train_indices,np.newaxis,:]

# Preprocesses testing
# SPLITS
X_test_raw = raw_test_sequence[:-1]
y_test_raw = raw_test_sequence[1:]
# ENCODES
X_test = pre.one_hot_encode_seq(X_test_raw, n_chars=len(int2char))
y_test = pre.one_hot_encode_seq(y_test_raw, n_chars=len(int2char))
# SHUFFLES
test_indices = pre.shuffle_indices(len(X_test_raw), BATCH_SIZE)
X_test = X_test[test_indices, np.newaxis,:]
y_test = y_test[test_indices]

print "Training data:"
print "X:", X_train.shape
print "y:", y_train.shape

print "Test data:"
print "X:", X_test.shape
print "y:", y_test.shape

# Recovers the original training data
X_test_sequence = test_sequence[:X_test.shape[0]]
y_test_sequence = test_sequence[1:X_test.shape[0]+1]
print 'Test sequences:', X_test_sequence[:5], y_test_sequence[:5]
print 'length:', len(X_test_sequence), len(y_test_sequence)


# # Builds model

# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,SimpleRNN
from keras.layers import Lambda
from keras import regularizers
from keras.optimizers import RMSprop,Adam

from controllers.mylstm_legacy import MYLSTM

in_dim = X_train.shape[1:]
out_dim = y_train.shape[1]

model = Sequential()
model.add(MYLSTM(N_STATES, 
                 stateful=True,
                 batch_size=BATCH_SIZE,
                 input_shape=in_dim,
                 dropout=.3))
model.add(Dense(out_dim, activation='softmax'))

optimizer = Adam(lr=0.000005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# In[11]:


from keras.models import load_model
from controllers.mylstm_legacy import MYLSTM
from keras.callbacks import ModelCheckpoint

if not CACHED:
    if CONTINUE:
        model = load_model('models/' + MODEL_NAME, custom_objects={'MYLSTM' :MYLSTM})
    model.reset_states()
    checkpoint = ModelCheckpoint(FOLDER_OUT + '/model-{epoch:02d}-{acc:.2f}',
                                 monitor='val_loss')
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=N_EPOCHS,
                        verbose=1,
                        shuffle=False,
                        callbacks = [checkpoint])
    model.save('models/' + MODEL_NAME )
    model.save('models/' + MODEL_NAME + '.bckup')
else:
    model = load_model('models/' + MODEL_NAME, custom_objects={'MYLSTM' :MYLSTM})


# # Evaluate

# In[12]:


model.reset_states()
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[13]:


# Generate samples
import generator as gen
reload(gen)

print gen.complete_sentence_stateful('if ', model, 256, char2int, int2char, BATCH_SIZE)
print '*****'
print gen.complete_sentence_stateful('else', model, 256, char2int, int2char, BATCH_SIZE)


# # Extracts features

# In[14]:


import features as feat
reload(feat)

#alphabet = feat.alphabet()

brackets = feat.is_enclosed_bin('(', ')')
curly_brackets = feat.is_enclosed_bin('{', '}')
linepos = feat.line_char_pos_bin()
comments = feat.is_enclosed_str_nodepth('/*','*/')
quotes = feat.is_enclosed_str_nodepth('"', '"')
indent = feat.line_indent_level()
word_hash = feat.char_word_hash(5)
alphanum = feat.alphanum()
word_limits = feat.word_start_end()

r1 = feat.regex('return')
r2 = feat.regex('struct')
r3 = feat.regex('e')
r4 = feat.regex('\d')
r5 = feat.regex('\s')
r6 = feat.regex('\n')
r7 = feat.regex('if\s*\(.*\)')
r8 = feat.regex('for\s*\(.*\)')

features = [linepos, brackets, curly_brackets, comments, 
            indent, word_hash, quotes, alphanum, word_limits,
            r1, r2, r3, r4, r5, r6, r7, r8]


# In[15]:


reload(feat)

feature_frame_x = feat.FeatureFrame(features, X_test_sequence)
feature_frame_x.extract()

feature_frame_y = feat.FeatureFrame(features, y_test_sequence)
feature_frame_y.extract()

print'Features for test sequence:'
print feature_frame_x.names
print feature_frame_x.values[:25,:]

print 'Features for test sequence labels:'
print feature_frame_y.names
print feature_frame_y.values[:25,:]


# # Extracts hidden states

# In[16]:


import extractor
reload(extractor)
    
ex=extractor.Extractor(model, LAYERS_TO_CHECK)
states = ex.get_states(X_test, batch_size=BATCH_SIZE, unshuffle=True)
nn_config = ex.get_structure()
nn_offsets = ex.get_offets()

print 'states shape:', states.shape
print 'config:', nn_config
print 'offets:', nn_offsets


# In[17]:


feat_names, feat_mat = feature_frame_x.data


# In[18]:


#get_ipython().run_cell_magic(u'R', u'-i feat_names,feat_mat,states,X_test_sequence,FOLDER_OUT', u'\nlibrary(ggplot2)\nlibrary(scales)\nlibrary(dplyr)\nlibrary(tidyr)\n\nstates_col_names <- paste0(\'_\', 0:(ncol(states)-1))\ncolnames(states) <- states_col_names\ncolnames(feat_mat) <- feat_names\n\nto_plot <- cbind(states, feat_mat)\nto_plot <- as.data.frame(to_plot)\nto_plot[[\'time\']] <- 1:nrow(to_plot)\nto_plot <-  gather(to_plot, key=\'Series\', value=\'Value\', -time)\nto_plot[[\'is_a_feature\']] <- ! to_plot$Series %in% states_col_names\n\nto_plot <- filter(to_plot, Series %in%states_col_names | is_a_feature)\nMAX_TIME <- 150 \nto_plot <- filter(to_plot, time <= MAX_TIME)\n\nlabels <- X_test_sequence[1:MAX_TIME]\nprint(labels)\n\np <- ggplot(to_plot, aes(x=time, y=Value, fill=is_a_feature, color=is_a_feature, shape = is_a_feature)) +\n            scale_x_continuous(breaks = sort(unique(to_plot$time)), labels = labels) +\n            #scale_y_continuous(limits = c(-1,1), breaks=c(-1,1)) +\n            geom_line() +\n            geom_point() +\n            facet_grid(Series~., scales="free")\n\nggsave(paste0(FOLDER_OUT, \'/activations.pdf\'), p, width=40, height=49)\n\nto_plot <- NULL\nstates <- NULL\nfeat_mat <- NULL\n\np')


# In[19]:


get_ipython().run_cell_magic(u'R', u'-i feat_names,feat_mat,states,X_test_sequence,FOLDER_OUT', u"\nlibrary(ggplot2)\nlibrary(scales)\nlibrary(dplyr)\nlibrary(tidyr)\n\nprint(X_test_sequence[1:5])\n\nstates_col_names <- paste0('_', 0:(ncol(states)-1))\ncolnames(states) <- states_col_names\ncolnames(feat_mat) <- feat_names\n\nto_plot <- cbind(states, feat_mat)\nto_plot <- as.data.frame(to_plot)\nto_plot[['time']] <- 1:nrow(to_plot)\nto_plot[['text']] <- X_test_sequence[1:nrow(to_plot)]\n\nto_plot <-  gather(to_plot, key='Series', value='Value', -time, -text)\n\nMAX_TIME <- 3000 \nto_plot <- filter(to_plot, time <= MAX_TIME)\n\nto_plot$Value <- round(to_plot$Value, 2)\nwrite.csv(to_plot, paste0(FOLDER_OUT, '/activations.csv'), row.names = FALSE)\n\nto_plot <- NULL\nstates <- NULL\nfeat_mat <- NULL")


# # Correlation based attribution

# In[20]:


import scores
reload(scores)
import inspector as ip
reload(ip)

insp = ip.Inspector(nn_config, nn_offsets)
mi_scores, names = insp.inspect(states, feature_frame_x, scores.Correlation())


# In[21]:


fname = FOLDER_OUT + '/attributions'
hnames = [repr(n) for n in names]
header = ','.join(hnames)
np.savetxt(fname, mi_scores, delimiter=',', header=header, comments='')


# In[22]:


get_ipython().run_cell_magic(u'R', u'-i mi_scores', u'values <- c(mi_scores)\nhist(values, breaks=30)')


# In[23]:


feature_neurons = insp.filter_attributions(ip.filter_threshold_abs(.1))
non_feature_neurons = insp.not_attributed()
feature_neurons


# In[24]:


is_selected = np.zeros_like(mi_scores)
for j, fname in enumerate(names):
    for adress in feature_neurons[fname]:
        i = insp.address_to_column(*adress)
        is_selected[i,j] = 1


# In[25]:


get_ipython().run_cell_magic(u'R', u'-i mi_scores,names,is_selected,N_STATES,FOLDER_OUT', u'\nlibrary(ggplot2)\nlibrary(scales) \nlibrary(dplyr)\nlibrary(tidyr)\n\n# Gets scores\ndata <- as.data.frame(mi_scores)\nnames(data) <- names\ndata[[\'Neuron\']] <- factor(0:(nrow(data)-1),\n                           levels = 0:(nrow(data)-1),\n                           labels = as.character(0:(nrow(data)-1)))\ndata <- gather(data, key=\'Feature\', value=\'Score\', -Neuron)\n\n# Gets neuron selection\nsel <- as.data.frame(is_selected)\nnames(sel) <- names\nsel[[\'Neuron\']] <- factor(0:(nrow(sel)-1),\n                           levels = 0:(nrow(sel)-1),\n                           labels = as.character(0:(nrow(sel)-1)))\nsel <- gather(sel, key=\'Feature\', value=\'selected\', -Neuron)\nsel$selected <- ifelse(sel$selected == 1, \'X\', \'\')\n\n# joins\nto_plot <- inner_join(data,sel, by = c("Neuron", "Feature"))\n\np <- ggplot(to_plot, aes(x=Feature, y=Neuron, fill=Score, label=selected)) + \n                geom_bin2d(aes=\'identity\') +\n                geom_text(color=\'red\') +\n                scale_fill_gradient2(midpoint=median(to_plot$Score),\n                                    low = muted("blue"), mid = "white",high = muted("red"),\n                                    limits=c(0,NA)) +\n                theme(axis.text.x = element_text(angle=90))\n\nggsave(paste0(FOLDER_OUT, \'/attribution_map.pdf\'), p, width=40, height=40)\n\nmi_scores <- NULL\nis_selected <- NULL\ndata <- NULL\nsel <- NULL\nto_plot <- NULL\n\np')


# # Does the Unit Test

# In[ ]:


import scores
print 'Scoring! Baseline'
out_base = insp.test(states, feature_frame_x, non_feature_neurons, scores.LogRegF1())
out_base


# In[ ]:


# With the correct neuronss 
print 'Scoring! Real neurons'
out = insp.test(states, feature_frame_x, feature_neurons, scores.LogRegF1())
out


# In[ ]:


# Prepares and saves a df
testnames = out.keys()
baseline = [out_base[t] for t in testnames]
candidate = [out[t] for t in testnames]


# In[ ]:


get_ipython().run_cell_magic(u'R', u'-i testnames,baseline,candidate,FOLDER_OUT', u"\ntoplot <- data.frame(testnames, baseline, candidate)\nwrite.csv(toplot, paste0(FOLDER_OUT,'/test_results'),row.names = FALSE)\n\ndat <- gather(toplot, key='Setting', value='Accuracy', -testnames)\n\np <- ggplot(dat, aes(x=testnames, y=Accuracy, fill=Setting, color=Setting)) +\n        geom_bar(stat='identity', position='dodge') +\n        theme(axis.text.x = element_text(angle=90))\n\nprint(p)\nggsave(paste0(FOLDER_OUT, '/test_scores.pdf'), p, width=40, height=10)")


# # Kevinizes

# In[ ]:


# from controllers import control
# reload(control)
# import verify
# reload(verify)

# verifier = verify.Verifier(model, feature_neurons, non_feature_neurons)
# v_scores = verifier.run(feature_frame_y, X_test, y_test, BATCH_SIZE, sample_size=SAMPLES_VERIFY)


# In[ ]:


# for feat in v_scores:
#     print '---', feat
#     for setup in v_scores[feat]:
#         print '-', setup
#         for y in v_scores[feat][setup]:
#             s = v_scores[feat][setup][y]
#             print y, ':', 'mean:', np.mean(s), '- sd:', np.std(s) 


# In[ ]:


# verifier.test_diff()


# In[ ]:


# import pandas as pd

# out = None

# for feat in v_scores:
#     print '---', feat
#     for setup in v_scores[feat]:
#         print '-', setup
#         for y in v_scores[feat][setup]:
#             s = np.array(v_scores[feat][setup][y])
#             print y, ':', 'mean:', np.mean(s), '- sd:', np.std(s) 
#             s_y = np.repeat(y, len(s))
#             s_setup = np.repeat(setup, len(s))
#             s_feat = np.repeat(feat, len(s))
#             df = pd.DataFrame({
#                 'acc' : s,
#                 'feat_val': s_y,
#                 'feature': s_feat,
#                 'setup' : s_setup
#             })
#             if out is None:
#                 out = df
#             else:
#                 out = pd.concat([out, df], axis = 0)

# print out
# out.to_csv(FOLDER_OUT+'/kevinizers.csv',index=False)


# # Post experiment checks

# In[ ]:


feat_names = feature_frame_x.names
feat_mat = feature_frame_x.values


# In[ ]:


get_ipython().run_cell_magic(u'R', u'-i feat_names,X_test_sequence,feat_mat,states', u'\nFEATURE <- c(\'char_i\')\nNEURON <- c(\'_156\')\n\nMIN_TIME <- 0\nMAX_TIME <- 1000\n\nlibrary(ggplot2)\nlibrary(scales)\nlibrary(dplyr)\nlibrary(tidyr)\n\nstates_col_names <- paste0(\'_\', 0:(ncol(states)-1))\ncolnames(states) <- states_col_names\ncolnames(feat_mat) <- feat_names\n\nto_plot <- cbind(states, feat_mat)\nto_plot <- as.data.frame(to_plot)\nto_plot[[\'time\']] <- 1:nrow(to_plot)\nto_plot <-  gather(to_plot, key=\'Series\', value=\'Value\', -time)\nto_plot[[\'is_a_feature\']] <- ! to_plot$Series %in% states_col_names\n\n\nto_plot <- filter(to_plot, Series %in% states_col_names | is_a_feature)\nto_plot <- filter(to_plot, Series %in% c(FEATURE, NEURON))\nto_plot <- filter(to_plot, time >= MIN_TIME, time <= MAX_TIME)\n\nlabels <- X_test_sequence[MIN_TIME:MAX_TIME]\n\n\np <- ggplot(to_plot, aes(x=time, y=Value, fill=is_a_feature, color=is_a_feature, shape = is_a_feature)) +\n            scale_x_continuous(breaks = sort(unique(to_plot$time)), labels = labels) +\n            #scale_y_continuous(limits = c(-1,1), breaks=c(-1,1)) +\n            geom_line() +\n            geom_point() +\n            facet_grid(Series~., scales="free")\nprint(p)\n\nfi <- paste0(\'~/Desktop/focus\', paste0(FEATURE, collapse=""), paste0(NEURON, collapse=""), \'.pdf\')\nh <- length(FEATURE) + length(NEURON)\nggsave(fi, p, width=40, height=h)')

