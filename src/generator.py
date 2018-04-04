import numpy as np

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def complete_sentence(sentence, model, nchars, char2int, int2char):
    out = np.zeros((1,len(sentence),len(int2char)))
    for i,s in enumerate(sentence):
        out[0,i,char2int[s]] = 1
    print 'Dimensions of sentence after encoding:', out.shape

    i = 0
    w = out.shape[1]
    while i < nchars:
        pred_v = model.predict(out[:,i:i+w,:])
        pred_c = sample(pred_v[0,])
        new_v = np.zeros_like(pred_v)
        new_v[0,pred_c] = 1
        new_v = np.expand_dims(new_v, axis=0)
        out = np.append(out, new_v, axis=1)
        i += 1

    print 'Dimensions of sentence after generation:', out.shape
    out = out[0,:,:]
    char_indices = np.argmax(out, axis=1)
    final_str = [int2char[int(i)] for i in char_indices]
    final_str = ''.join(final_str)
    return(final_str)


def complete_sentence_stateful(sentence, model, nchars, char2int, int2char, batch_size):
    out = np.zeros((batch_size,len(sentence),len(int2char)))
    for i,s in enumerate(sentence):
        out[:batch_size,i,char2int[s]] = 1
    print 'Dimensions of sentence after encoding:', out.shape

    model.reset_states()
    for i in range(out.shape[1] + nchars):
        if i < out.shape[1]-1:
            continue
        pred_v = model.predict(out[:,i:i+1,:])
        #print 'Just read character:', int2char[np.argmax(out[0,i,:])]
        # print 'Predictions:' , pred_v.shape
        # print pred_v

        pred_c = sample(pred_v[0,])
        new_v = np.zeros((out.shape[0],1,out.shape[2]))
        new_v[:,0,pred_c] = 1
        out = np.append(out, new_v, axis=1)

    print 'Dimensions of sentence after generation:', out.shape
    out = out[0,:,:]
    char_indices = np.argmax(out, axis=1)
    final_str = [int2char[int(i)] for i in char_indices]
    final_str = ''.join(final_str)
    return(final_str)