#! /usr/bin/env python
'''
Train LSTM-RNN to predict next word.
Built from trainEmbedSpot.  This version does NOT use
word2vec vectors.

USAGE: testKeras MODEL
'''
import sys
import os
import time
import math
import numpy as np
from utilsKeras import *
from datetime import datetime,timedelta

import bz2
import cPickle as pickle
from subprocess import call
#  Keras
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Activation, SimpleRNN, LSTM, Embedding

from keras.optimizers import RMSprop
from keras import initializers

import keras.backend as K

def evalPPL(pred,actual,vlen,start,end):
  '''Return prob of predicted outputs based on actual desired outputs.'''

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


MODEL_INPUT_FILE = sys.argv[1]
DEBUG = False # Debug version for quicker testing
INPUT_DATA_FILE = "./data/test/NoOverlapRawTest.pbz2"

## Input Data
with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
  (voc,sentences) = pickle.load(handle)

VOCABULARY_SIZE = len(voc)
INPUT_DIM = VOCABULARY_SIZE
OUTPUT_DIM = VOCABULARY_SIZE

inputs = np.asarray( [[w+1 for w in sent[:-1]] for sent in sentences])
outputs = np.asarray( [[w+1 for w in sent[1:]] for sent in sentences])

maxlen = 108 # taken from training inputs
print 'Feature maxlen',maxlen

for i in xrange(len(outputs)):
    for j in xrange(len(outputs[i])):
        outputs[i][j] = [outputs[i][j]]

inputs_len = map(len,inputs)

inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=maxlen)
outputs = keras.preprocessing.sequence.pad_sequences(outputs, maxlen=maxlen)


print "Vocab size %d" % VOCABULARY_SIZE
print "Output dim %d" % OUTPUT_DIM
print 'shape of inputs', inputs.shape
print 'shape of outputs', outputs.shape


# Build model =============================
print 'Loading saved Network...'
model = keras.models.load_model(MODEL_INPUT_FILE)
model.summary()


#scores = model.evaluate(inputs,outputs,batch_size=100,verbose=0)
#print 'Validation score',scores[0]
#print 'Validation accuracy',scores[1]
          
logprob,wcount = 0.0,0
predMax = 500
start = 0
Nsent = len(inputs)
while start < Nsent:
  end = min(Nsent,start + predMax)
  pred = model.predict(inputs[start:end],verbose=0,batch_size=100)
  lp,wc = evalPPL(pred,outputs,inputs_len,start,end)
  logprob += lp
  wcount += wc
  start += predMax
  
print '\nInput data file: %s' % INPUT_DATA_FILE
print 'logprob %f wcount %f entropy %f' % (logprob,wcount,-logprob/wcount)
print 'PPL %f' % pow(2.0,-logprob/wcount)


