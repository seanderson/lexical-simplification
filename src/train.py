#! /usr/bin/env python
'''
Train LSTM-RNN to predict next word.
Built from trainEmbedSpot.  This version does NOT use
word2vec vectors.

USAGE: train MODEL DATE
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
from keras.layers import Dense, Activation,SimpleRNN,Embedding
from keras.optimizers import RMSprop
from keras import initializers

import keras.backend as K


DEBUG = True  # Debug version for quicker testing
INPUT_DATA_FILE = "../data/train/NoOverlapRawTrain.pbz2"
VALID_DATA_FILE = "../data/train/NoOverlapRawValid.pbz2"
# INPUT_DATA_FILE = VALID_DATA_FILE  # Is it for debugging??

HIDDEN_DIM = 100
LEARNING_RATE = 0.1
EMBEDDING_DIM = 1000
NEPOCH = 100

if len(sys.argv) > 1:
    MODEL_INPUT_FILE = sys.argv[1]
else:
    MODEL_INPUT_FILE = "INIT"

if DEBUG:
  print 50*'=','\nDEBUGGING = TRUE\n',50*'='
  # INPUT_DATA_FILE = "../data/test/DebugRawTest.pbz2"
  # VALID_DATA_FILE = "../data/test/DebugRawTest.pbz2"
  NEPOCH = 6
  HIDDEN_DIM = 10
  EMBEDDING_DIM = 10
  MODEL_OUTPUT_FILE = ""


LAST_UPDATE=datetime.now()

## Training Data
with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
  (voc,sentences) = pickle.load(handle)

VOCABULARY_SIZE = len(voc)
INPUT_DIM = VOCABULARY_SIZE
OUTPUT_DIM = VOCABULARY_SIZE

inputs = np.asarray( [[w+1 for w in sent[:-1]] for sent in sentences])
outputs = np.asarray( [[w+1 for w in sent[1:]] for sent in sentences])
if DEBUG:
  inputs = inputs[:1000]
  outputs = outputs[:1000]

maxlen = max(map(len,inputs))

for i in xrange(len(outputs)):
    for j in xrange(len(outputs[i])):
        outputs[i][j] = [outputs[i][j]]

inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=maxlen)
outputs = keras.preprocessing.sequence.pad_sequences(outputs, maxlen=maxlen)

print "Embedding dim %d" % EMBEDDING_DIM
print "Vocab size %d" % VOCABULARY_SIZE
print "Output dim %d" % OUTPUT_DIM
print 'shape of inputs', inputs.shape
print 'shape of outputs', outputs.shape

## Validation Data
with bz2.BZ2File(VALID_DATA_FILE,'r') as handle:
  (voc_v,sentences_v) = pickle.load(handle)


inputs_v = np.asarray( [[w+1 for w in sent[:-1]] for sent in sentences])
outputs_v = np.asarray( [[w+1 for w in sent[1:]] for sent in sentences])
if DEBUG:
  inputs_v = inputs_v[:1000]
  outputs_v = outputs_v[:1000]

for i in xrange(len(outputs_v)):
    for j in xrange(len(outputs_v[i])):
        outputs_v[i][j] = [outputs_v[i][j]]

inputs_v = keras.preprocessing.sequence.pad_sequences(inputs_v, maxlen=maxlen)
outputs_v = keras.preprocessing.sequence.pad_sequences(outputs_v, maxlen=maxlen)

print 'Num Validation Sentences: %d' % len(inputs_v)


# Build model =============================

if MODEL_INPUT_FILE == 'INIT':
  print 'Building Network...'
  model = Sequential()

  model.add(Embedding(INPUT_DIM+1,
                      EMBEDDING_DIM,
                      input_length=None,
                      embeddings_initializer = 'uniform',
                      activity_regularizer=None,
                      embeddings_constraint=None,
                      mask_zero=True ) )

  model.add( SimpleRNN(HIDDEN_DIM,
                       recurrent_initializer=initializers.Identity(gain=1.0),
                       return_sequences=True,
                       activation='tanh',
                       dropout=0.01) )

  model.add(Dense(OUTPUT_DIM,activation='softmax'))
  
  rmsprop = RMSprop(lr=LEARNING_RATE)
  
  # For a multi-class classification problem
  model.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
else:
  print 'Loading saved Network...'
  model = keras.models.load_model(MODEL_INPUT_FILE)


model.summary()


filepath="../srn-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='min',
                                             period=5)

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=1,
                                              verbose=1,
                                              mode='auto')

callbacksList = [checkpoint,earlyStopping]


model.fit(inputs,outputs,
          batch_size=100,
          epochs=NEPOCH,
          verbose=1,
          callbacks=callbacksList,
          validation_data=(inputs_v,outputs_v))


scores = model.evaluate(inputs_v,outputs_v,verbose=0,batch_size=100)
print 'Validation score',scores[0]
print 'Validation accuracy',scores[1]
          
