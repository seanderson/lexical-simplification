#! /usr/bin/env python

from globdefs import *
import theano
import bz2
import csv
import itertools
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordPOSTagger
from nltk.corpus import wordnet
import time
import sys
import operator
import io
import array
from datetime import datetime
from gru_theano_embed import GRUTheano
import cPickle as pickle


def lemmatize(tok_sentences):
    '''Lemmatize list of tokenized sentences.'''
    morphy_tag = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}
    tagger = StanfordPOSTagger(STANFORD_MODEL,STANFORD_JAR)
    lm = WordNetLemmatizer()

    def lemm(word,pos):
        wnetpos=morphy_tag.get(pos[:2],None)
        if wnetpos == None: return lm.lemmatize(word) # use POS if possible
        return lm.lemmatize(word,wnetpos)
    tagged_sentences = tagger.tag_sents(tok_sentences)
    lemmatized_sentences = [ ]
    for s in tagged_sentences:
        tagged_s = [ lemm(w[0],w[1]) for w in s ]
        lemmatized_sentences.append(tagged_s)
    
    return lemmatized_sentences

def readFile(filename):
    # Read the data and append SENT_START and SENTENCE_END tokens
    print("Reading file..."+ filename)
    with open(filename, 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True,delimiter='\t')
        # tab delim line format: sentence\tword\tpositition\tcomplex_flag
        lines = [x for x in reader]
        sentences = map( lambda x: x[0].decode("utf-8").lower() , lines)
        outputs = map( lambda x: int(x[3]), lines)
        words = map( lambda x: x[1].decode("utf-8").lower() , lines)
        positions = map( lambda x: int(x[2]), lines)
    print("Parsed %d sentences." % (len(sentences)))
    return (sentences,outputs,words,positions)

def tokenizeSentences(tokfile,sentences):
    '''Load tokenized sentences, creating if necessary.
    Each sentence might have a word that should also be lemmatized, using
    the same part of speech.'''
    tokenized_sentences = [ ]    
    # Get from bz2 pickle
    if tokfile != None:
        print('loading ',tokfile)
        with bz2.BZ2File(tokfile,'r') as handle:
            tokenized_sentences = pickle.load(handle)
    else:  # tokenize/lemmatize/save sentences
        tokenized_sentences = [ nltk.word_tokenize(sent) for sent in sentences ]
        tokenized_sentences = lemmatize(tokenized_sentences)
        with bz2.BZ2File('tok_sentences.pbz2','w') as handle:
            pickle.dump(tokenized_sentences,handle)
    return tokenized_sentences

def preprocess_data_embed(filename="data/train/cwi_training.txt", min_sent_characters=0, tokfile=None, embedfile="data/word2veclex.pbz2"):
    '''Tokenize sentences, lemmatize, prepare network data, use embedded vecs.
    filename: list of training sentences (from Semeval16)
    tokfile: tokenized/lemmatized sentences
    embedfile: file of word->vector mappings
    '''
    word_to_index = []
    index_to_word = []

    # Load Word2Vec vectors
    with bz2.BZ2File('data/word2veclex.pbz2','r') as handle:
        wordvecs = pickle.load(handle)
    vocabsize = len(wordvecs) + NUM_SPECIAL_WORDS
    veclen = len(wordvecs['cat']) + NUM_SPECIAL_WORDS
    
    keys = [SENT_START,WORD_UNKNOWN,WORD_NUMERIC] + sorted(wordvecs.keys())
    vecs = np.zeros( (veclen,vocabsize) , dtype=theano.config.floatX)
    # code for special words not in vocabulary
    for i in range(NUM_SPECIAL_WORDS):
        vecs[i][i] = 1.0
    for i in range(NUM_SPECIAL_WORDS,vocabsize):
        if not isinstance(wordvecs[keys[i]],str):
            vecs[NUM_SPECIAL_WORDS:,i] = wordvecs[keys[i]]
        
    (sentences,outputs,words,positions) = readFile(filename)
    # Tokenize+lemmatize the sentences into words (from file)
    tokenized_sentences = tokenizeSentences(tokfile,sentences=None)
    
    # Add sentence start
    for i,t in enumerate(tokenized_sentences):
        t.insert(0,SENT_START)
        t.insert(0,t[1+positions[i]]) 
    word_to_index = dict([(w, i) for i, w in enumerate(keys)])

    # Create the training data
    inputs = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
    targets = np.asarray( [[x] for x in outputs])

    '''for j in range(10):
        r = np.random.randint(0,len(inputs))
        print tokenized_sentences[r]
        for i,widx in enumerate(inputs[r]):
            print keys[widx],
        print "\n-----"
    '''
    return inputs, targets, keys, vecs

# End of preprocess_data_embed

def preprocess_data(filename="data/train/cwi_training.txt", vocabulary_size=100, min_sent_characters=0, tokfile=None):
    '''Tokenize sentences, lemmatize, prepare network data.
    Do not use embedded vectors.  Embedding is learned.
    '''
    word_to_index = []
    index_to_word = []

    (sentences,outputs,words,positions) = readFile(filename)
    # Tokenize+lemmatize the sentences into words
    tokenized_sentences = tokenizeSentences(tokfile,sentences)
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))
    # Get the most common words and build index_to_word and word_to_index vectors
    # Save spots for Sentence markers and UNKNOWN vocab items
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN,SENT_START] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Add target word and sentence start
    for i,t in enumerate(tokenized_sentences):
        t.insert(0,SENT_START)
        t.insert(0,words[i])
    
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
    #y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    y_train = np.asarray( [[x] for x in outputs])

    return X_train, y_train, word_to_index, index_to_word



def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
    callback_every=10, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)            
    return model

def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value())
    print "Saved model parameters to %s." % outfile

def load_model_parameters_theano(path, modelClass=GRUTheano,wordEmbed=None):
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim,output_dim = E.shape[0], E.shape[1], V.shape[0]
    print "Building model model from %s with hidden_dim=%d word_dim=%d output_dim=%d" % (path, hidden_dim, word_dim, output_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim,output_dim=output_dim,wordEmbed=wordEmbed)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    return model 

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['E', 'U', 'W', 'b', 'V', 'c']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return 
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)


def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()

def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENT_START]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence

def generate_sentences(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence(model, index_to_word, word_to_index)
        print_sentence(sent, index_to_word)


def run_sentence2(model,x,y,topN=2):
    '''Run sentence to accumulate word prediciton probabilities.  Input at
    each step is provided by actual sentence.  Return complete list of
    arrays of probabilities from first actual word to end of sentence.
    '''
    allprobs = [ ]
    new_sentence = [ ]
    logppl = 0.0 # perplexity
    for i in range(0,len(x)-1):
        new_sentence.append(x[i])
        next_word_probs = model.predict(new_sentence)[-1]
        ind = np.argpartition(next_word_probs,-topN)[-topN:] # topN prob indices
        vals = next_word_probs[ind]
        allprobs.append( (ind,vals) )
    return allprobs

def run_sentence(model,x,y,topN=2):
    '''Run sentence to get word prediciton probabilities.  Input at
    each step is provided by actual sentence.  Return complete list of
    arrays of probabilities from first actual word to end of sentence.
    '''
    next_word_probs = model.predict(x)
    logppl = 0.0
    n = len(y)
    # last col is actual next word index/prob
    iarr = np.ndarray( (n,topN+1), np.int32)
    parr = np.ndarray( (n,topN+1), theano.config.floatX)

    for i,p in enumerate(next_word_probs): # each word in sentence
        logppl += np.log2( p[y[i]] ) # logprob of actual next word
        ind = np.argpartition(p,-topN)[-topN:] # top ten prob indices
        vals = p[ind]
        np.copyto(iarr[i][:-1],ind)
        np.copyto(parr[i][:-1],vals)
        iarr[i][topN] = y[i]
        parr[i][topN] = p[y[i]]

    return iarr,parr,logppl

def writeprobs(fout,arr1,arr2):
    '''Write all data for one sentence, delimited by newlines.'''
    dat = arr1.flatten().tolist()
    n = len(dat)
    fout.write(str(n) + '\n')
    fout.write('\n'.join( map(str,dat) ) + '\n')
    dat = arr2.flatten().tolist()
    fout.write('\n'.join( map(str,dat) ) + '\n')


def readprobs(fd,ncol=3):
    '''Read all data for one sentence, delimited by newlines.
    For each sentence int data first, then floats.'''
    line = fd.readline()
    if line == '': return [],[]
    n = int(line)
    arr = np.ndarray( (n,1) , theano.config.floatX)
    iarr = np.ndarray( (n,1) , np.int32 )
    for i in range(n):
        iarr[i] = int(fd.readline())
    for i in range(n):
        arr[i] = float(fd.readline())
    return np.reshape(iarr, (n/ncol,ncol)), np.reshape(arr, (n/ncol,ncol))

def allWordProbs(model, inputs, outputs):
    '''Start with sentence start and collect probability of all predicted
    next words.  Returns fname file containing 
    list of lists, one for each sentence.  Each
    sentence list is unordered pairs with (indices,values) for topN
    word probs.
    '''
    fname = 'allprobs.pbz2'
    fout = bz2.BZ2File(fname,'w')
    logppl = 0.0
    numwords = 0
    Num =  len(outputs)
    topN = 2
    
    for i in range(Num):
        iarr,parr,logppl_s = run_sentence(model,inputs[i],outputs[i],topN=topN)
        numwords += len(outputs[i])
        logppl += logppl_s
        writeprobs(fout,iarr,parr)
    fout.close()
    return fname,logppl,numwords
        
def test_all(model, X, y):
    '''Score outputs vs. desired outputs.'''

    outputs = np.zeros(len(y))
    for i in range(len(y)):
        # One testing step
        cost,outputs[i] = model.test_step(X[i], y[i])
    return score(outputs,y)
        

def save_output(model, X, y, outfile='outputs.dat'):
    '''Dump actual/desired outputs, by sentence.'''
    # For each training example...
    outputs = np.zeros(len(y))
    for i in range(len(y)):
        # One testing step
        cost,outputs[i] = model.test_step(X[i], y[i])
    with open(outfile,'w') as fd:
        for i in range(len(y)):
            fd.write("%5.3f %2.1f\n" % (outputs[i],y[i]))
        

def score(output,desired):
    THRESH = 0.5
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(output)):
        if desired[i] >= THRESH:
            if output[i] >= THRESH: tp += 1
            else: fn += 1
        else:
            if output[i] < THRESH: tn += 1
            else: fp += 1
    return (tp,tn,fp,fn)


def wordStats(probs,indices,outputs,fout=""):
  '''return output data for each word in a sentence.
  col 1: 0/1 if it is correct
  col 2: first/second probability ratio
  col 3: actual output word (index in lexicon)
  col 4: index of greatest probability
  col 5: prob of actual predicted word
  probs is the probabilities for words in the sentence
  indices are word indices for probabilities
  outputs is the correct word indices for the sentence
  '''
  writeOutput = (fout != "")
  results = [ ]
  num = 0
  numcorrect = 0
  for i,p in enumerate(probs): # all words (each an array)
      num += 1
      maxidx = np.argmax(p[:-1])
      minidx = np.argmin(p[:-1])
      maxval = p[maxidx]
      idx = indices[i][maxidx]

      #print 'types',idx,type(idx),type(outputs[i]),outputs[i]
      #if i >= len(outputs):
      #    print 'outputs',outputs
      iscorrect = 0
      if idx == outputs[i]:
        iscorrect = 1
        numcorrect += 1
      msglst = (iscorrect, maxval/p[minidx],idx,outputs[i],p[-1])
      if writeOutput:
          fout.write("%d %f %d %d %f\n"  % msglst)
      else:
          results.append(msglst)
  return results
