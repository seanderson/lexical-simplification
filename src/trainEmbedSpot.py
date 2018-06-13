#! /usr/bin/env python
'''
Train LSTM-RNN to predict next word.
Uses word2vec vectors (len 300) for embedding.
USAGE: trainEmbedSpot MODEL DATE
'''
import sys
import os
import time
import math
import numpy as np
from utils import *
from datetime import datetime,timedelta
from gru_theano_embed import GRUTheano
import bz2
import cPickle as pickle
from subprocess import call

DEBUG = False # Debug version for quicker testing
SAVE_INTERVAL = 60
INPUT_DATA_FILE = "./data/train/NoOverlapTrain.pbz2"
VALID_DATA_FILE = "./data/train/NoOverlapValid.pbz2"
PRINT_EVERY = 10000
HIDDEN_DIM = 100
SAVE_TO_S3 = False
if DEBUG == True:
  SAVE_INTERVAL = 1
  INPUT_DATA_FILE = "../data/test/DebugTest.pbz2"
  VALID_DATA_FILE = "./data/test/DebugTest.pbz2"
  PRINT_EVERY = 200
  HIDDEN_DIM = 10
  SAVE_TO_S3 = False

USE_EMBEDDING = True # Use pre-made vocabulary vectors
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.0001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "0"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "0"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = ""
MODEL_INPUT_FILE = sys.argv[1]
LOGFILE=sys.argv[2]
LAST_UPDATE=datetime.now()

def outfilename():
  '''outfile name updates every 60 minutes.  Thus copied files only create
  new files only that often.'''
  global LAST_UPDATE
  ts=datetime.now() #.strftime("%Y-%m-%d-%H-%M")
  if ts - LAST_UPDATE > timedelta(minutes=SAVE_INTERVAL):
    LAST_UPDATE = datetime.now()
  tstring = LAST_UPDATE.strftime("%Y-%m-%d-%H-%M")
  return "GRU-%s-%s-%s-%s.dat" % (tstring, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

def logfile():
  global LOGFILE
  print 'logfile',LOGFILE
  return LOGFILE

## Training Data
with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
  (voc,vecs,sentences) = pickle.load(handle)

EMBEDDING_DIM = vecs.shape[0]
OUTPUT_DIM = vecs.shape[1]

inputs = np.asarray( [[w for w in sent[:-1]] for sent in sentences])
outputs = np.asarray( [[w for w in sent[1:]] for sent in sentences])

print "Embedding dim %d" % EMBEDDING_DIM
print "Vocab size %d" % OUTPUT_DIM
print "Output dim %d" % OUTPUT_DIM
print 'shape of inputs', inputs.shape
print 'shape of outputs', outputs.shape

## Validation Data
with bz2.BZ2File(VALID_DATA_FILE,'r') as handle:
  (voc_v,vecs_v,sentences_v) = pickle.load(handle)

inputs_v = np.asarray( [[w for w in sent[:-1]] for sent in sentences_v])
outputs_v = np.asarray( [[w for w in sent[1:]] for sent in sentences_v])
print 'Num Validation Sentences: %d' % len(inputs_v)

# Build model or load old one
if MODEL_INPUT_FILE == 'INIT':
  model = GRUTheano(word_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, bptt_truncate=-1,wordEmbed=vecs)
  #save_model_parameters_theano(model, 'INIT-' + outfilename())
else:
  model = load_model_parameters_theano(MODEL_INPUT_FILE,wordEmbed=vecs)

# Print SGD step time
t1 = time.time()
model.sgd_step(inputs[1], outputs[1], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

#sys.exit(1)
foo = 0
epoch = 1
Loss = 0.0
LossAvg = []
# We do this every PRINT_EVERY iterations.
def sgd_callback(model, num_examples_seen):
  global Loss, LossAvg,epoch, LEARNING_RATE
  dt = datetime.now().isoformat()
  Loss = model.calculate_loss(inputs_v, outputs_v)
  LossAvg.append(Loss)
  print("\n%s (%d) epoch %d" % (dt, num_examples_seen,epoch))
  print("--------------------------------------------------")
  print("Loss: %f" % Loss)
  print("\n")
  sys.stdout.flush()
  if math.isnan(Loss): # if NAN we need to stop everything.
    print "Error NaN.  Terminating."
    if SAVE_TO_S3:
	call( ["s3cmd","xput",logfile(),"s3://rnn-1/wpred/"] )
    	call(["./selfTerminate.sh"]) # terminates entire EC2
    sys.exit(-1) # should never get here unless selfterminate fails.
  # save file and send to S3
  fname = outfilename()
  save_model_parameters_theano(model, fname)
  fname = fname + ".npz"
  if SAVE_TO_S3:
    #call( ["s3cmd","put","--signature-v2",fname,"s3://rnn-1/wpred/"] )
    print 'saving',logfile()
    call( ["s3cmd","put","--signature-v2",logfile(),"s3://rnn-1/wpred/"] )
  if (num_examples_seen/len(inputs) > epoch):
    epoch += 1
    print 'Epoch: %f' % epoch

    midpt = len(LossAvg)/2
    if sum(LossAvg[0:midpt])/float(midpt) <= sum(LossAvg[midpt:])/float(len(LossAvg)-midpt):
      LEARNING_RATE /= 2.0
      print "New learning rate: %f" % LEARNING_RATE
    LossAvg = []    
    
train_with_sgd(model, inputs, outputs, learning_rate=LEARNING_RATE, nepoch=NEPOCH, decay=0.9, callback_every=PRINT_EVERY, callback=sgd_callback)


