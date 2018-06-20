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

import classpaths as path
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

import h5py
import utils_for_reno_kriz_data

DEBUG = False
DEBUG_SIZE = 11

MODEL_INPUT_NAME = sys.argv[1]
MODEL_INPUT_FILE = path.NN_MODELS + MODEL_INPUT_NAME
# INPUT_DATA_FILE = "./data/test/NoOverlapRawTest.pbz2"
# INPUT_DATA_FILE = "./data/test/NewselaSimple03test.pbz2"
INPUT_DATA_FILE = path.nnetFile

## Input Data
with bz2.BZ2File(utils_for_reno_kriz_data.bz2_file, 'r') as handle:
  # (voc,sentences) = pickle.load(handle)
  (voc
   , sentences) = pickle.load(handle)

VOCABULARY_SIZE = len(voc)
INPUT_DIM = VOCABULARY_SIZE
OUTPUT_DIM = VOCABULARY_SIZE

inputs = np.asarray( [[w+1 for w in sent[:-1]] for sent in sentences])
inputs_len = map(len,inputs)
#print("length: ", str(len(inputs)))
if DEBUG: print (inputs_len[:DEBUG_SIZE])

outputs = np.asarray( [[w+1 for w in sent[1:]] for sent in sentences])


maxlen = 108 # taken from training inputs
print ('Feature maxlen',maxlen)

for i in xrange(len(outputs)):
    for j in xrange(len(outputs[i])):
        outputs[i][j] = [outputs[i][j]]




inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=maxlen)
outputs = keras.preprocessing.sequence.pad_sequences(outputs, maxlen=maxlen)
if DEBUG:
  inputs = inputs[:DEBUG_SIZE]
  outputs = outputs[:DEBUG_SIZE]

print ("Vocab size %d" % VOCABULARY_SIZE)
print ("Output dim %d" % OUTPUT_DIM)
print ('shape of inputs', inputs.shape)
print ('shape of outputs', outputs.shape)


# Build model =============================
print ('Loading saved Network...')
model = keras.models.load_model(MODEL_INPUT_FILE)
model.summary()

if DEBUG:
  probsfile = path.PREDICTIONS + MODEL_INPUT_NAME[:-5] + '-'+str(DEBUG_SIZE)+'probs.h5'
else:
  # probsfile = path.PREDICTIONS + MODEL_INPUT_NAME[:-5] + '-probs.h5'
  probsfile = path.PREDICTIONS + 'paper-probs.h5'


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
  lp = 100 # hack!
  wc = 100 # hack!
  while start < Nsent:
    print("new_while_loop")
    end = min(Nsent,start + predMax)
    pred = model.predict(inputs[start:end],verbose=0,batch_size=100)
    #print 'pred shape',pred.shape
    #lp,wc = evalPPL(pred,outputs,inputs_len,start,end)
    # each sentence is one matrix in outfile
    for sentpred in pred:
      dat = writeProbs(sentpred, outputs[start], inputs_len[start])
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
if DEBUG:
    h5fd = h5py.File(probsfile,'r')
    sents = readProbs(h5fd)
    for i in range(DEBUG_SIZE):
        sent = sents[i]
        print(' '.join([voc[int(x[0][1])-1] for x in sent]))
    h5fd.close()
