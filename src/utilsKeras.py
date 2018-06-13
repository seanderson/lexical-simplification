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

import cPickle as pickle
import h5py

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


def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()



def score(output,desired):

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
      #print 'idx',idx,i,outputs
      if idx == outputs[i]:
        iscorrect = 1
        numcorrect += 1
      msglst = (iscorrect, maxval/p[minidx],idx,outputs[i],p[-1])
      if writeOutput:
          fout.write("%d %f %d %d %f\n"  % msglst)
      else:
          results.append(msglst)
  return results


def evalPPL(pred,actual,vlen,start,end):
  '''Return prob of predicted outputs based on actual desired outputs
  from actual[start:end]
  pred predicted probs from network (via model.predict) from start to end
  actual index of actual correct word for ALL inputs
  vlen list of lengths (numwords) for ALL input sentences
  start starting index for evaluation of perplexity
  end final index for eval of perplexity
  '''
  logprob = 0.0
  wcount = 0
  maxlen = len(pred[0])

  for i in xrange(start,end):
    startpos = maxlen - vlen[i]
    wcount += maxlen - startpos
    for j in xrange(startpos,maxlen):
      actualword = actual[i][j][0]
      prob = pred[i-start][j][int(actualword)]
      logprob += np.log2(prob)
  return logprob,wcount


def writeProbs(pred,actual,n):
    '''Appends topN probabilities for a single sentence in pred 
    to h5fd file in a single numpy 3-d array nword arrays, each of 
    which has topN+1 (probability,index) pairs.
    pred is network predictions for one sentence
    actual is actual word index for all words in sentence
    n is number of words in sentence
    '''
    topN = 3  # num probabilities to save (also saves for word of greatest prob.)
    maxlen = len(pred)
    startpos = maxlen - n # skips initial zero padding from keras
    dat = np.zeros( (n,topN+1,2), np.float32)
    for j in xrange(startpos,maxlen): # jth word in sentence
        jp = j - startpos
        p = pred[j] # probs for word j
        actualIdx = actual[j][0]
        ind = np.argpartition(p,-topN)[-topN:] # topN prob indices
        vals = p[ind] # prob vals of topN words
        indvals = np.stack( (vals,ind), axis=-1)
        indvals = np.array(sorted(indvals,key=lambda tup: tup[0],reverse=True))
        a = np.array( [[p[actualIdx],actualIdx]] )
        indvals = np.concatenate( (a,indvals), axis=0)
        np.copyto(dat[jp], indvals)  # the first tuple is the probability of teh correct word
    return dat


def readProbs(h5fd, snum=-1):
    '''Read all probs data for snum sentence and return as list with one
    array.  if snum == -1 all sentence data are returned as list of
    arrays.
    '''
    dat = [ ]
    if snum == -1:
        indexes = sorted([int(x) for x in h5fd])
        # because the indexes are strings they would be ordered
        # alphabeticaly by default which would cause errors. E.g. 
        # '10' would appear before '2'. This is why the indexes are
        # converted to int and are sorted only afterwards
        for name in indexes:
            dat.append( h5fd[str(name)][:] )
    else:
        dat.append( h5fd[str(snum)][:] )
    return dat

