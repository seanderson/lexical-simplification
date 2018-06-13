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
from utils import *
from datetime import datetime,timedelta


import bz2
import cPickle as pickle
from subprocess import call
#  Keras
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Activation,SimpleRNN,LSTM,Embedding
from keras.optimizers import RMSprop
from keras import initializers

import keras.backend as K

DEBUG = False

MODEL_INPUT_NAME = sys.argv[1]
MODEL_INPUT_FILE = path.NN_MODELS + MODEL_INPUT_NAME
# INPUT_DATA_FILE = "./data/test/NoOverlapRawTest.pbz2"
# INPUT_DATA_FILE = "./data/test/NewselaSimple03test.pbz2"
INPUT_DATA_FILE = path.nnetFile

## Input Data
with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
  (voc,sentences) = pickle.load(handle)

VOCABULARY_SIZE = len(voc)
INPUT_DIM = VOCABULARY_SIZE
OUTPUT_DIM = VOCABULARY_SIZE

inputs = np.asarray( [[w+1 for w in sent[:-1]] for sent in sentences])
inputs_len = map(len,inputs)
if DEBUG: print (inputs_len[:10])

outputs = np.asarray( [[w+1 for w in sent[1:]] for sent in sentences])

maxlen = 108 # taken from training inputs
print ('Feature maxlen',maxlen)

for i in xrange(len(outputs)):
    for j in xrange(len(outputs[i])):
        outputs[i][j] = [outputs[i][j]]




inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=maxlen)
outputs = keras.preprocessing.sequence.pad_sequences(outputs, maxlen=maxlen)

if DEBUG: # shorten i/o for debugging
  inputs = inputs[:10]
  outputs = outputs[:10]

print ("Vocab size %d" % VOCABULARY_SIZE)
print ("Output dim %d" % OUTPUT_DIM)
print ('shape of inputs', inputs.shape)
print ('shape of outputs', outputs.shape)


# Build model =============================
print ('Loading saved Network...')
model = keras.models.load_model(MODEL_INPUT_FILE)
model.summary()

probsfile = path.PREDICTIONS + MODEL_INPUT_NAME[:-5] + '-probs.h5'

if DEBUG:
  print (50*'=','\nDEBUGGING = TRUE\n',50*'=')
if 'eval' in sys.argv:
  scores = model.evaluate(inputs,outputs,batch_size=100,verbose=0)
  print ('Validation score',scores[0])
  print ('Validation accuracy',scores[1])
if 'ppl' in sys.argv or 'probs' in sys.argv:
  h5fd = h5py.File(probsfile, 'w')
  logprob,wcount = 0.0,0
  predMax = 500
  start = 0
  Nsent = len(inputs)
  while start < Nsent:
    end = min(Nsent,start + predMax)
    pred = model.predict(inputs[start:end],verbose=0,batch_size=100)
    #print 'pred shape',pred.shape
    lp,wc = evalPPL(pred,outputs,inputs_len,start,end)
    # each sentence is one matrix in outfile
    for sentpred in pred:
      dat = writeProbs(sentpred,outputs[start],inputs_len[start])
      h5fd.create_dataset(str(start), data=dat)
      start += 1
    logprob += lp
    wcount += wc
  h5fd.close()  
  if 'ppl' in sys.argv:
    print ('\nInput data file: %s' % INPUT_DATA_FILE)
    print ('logprob %f wcount %f entropy %f' % (logprob,wcount,-logprob/wcount))
    print ('PPL %f' % pow(2.0,-logprob/wcount))

# read probs file
#h5fd= h5py.File(probsfile,'r')
#print readProbs(h5fd,1)
#h5fd.close()
