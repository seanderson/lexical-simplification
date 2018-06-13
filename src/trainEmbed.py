#! /usr/bin/env python
'''
Train LSTM-RNN to predict next word.
Uses word2vec vectors (len 300) for embedding.
'''
import sys
import os
import time
import numpy as np
from utils_embed import *
from datetime import datetime
from gru_theano_embed import GRUTheano
import bz2
import cPickle as pickle

USE_EMBEDDING = True # Use pre-made vocabulary vectors

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.000005"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "0"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "0"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "100"))
NEPOCH = int(os.environ.get("NEPOCH", "3"))
MODEL_OUTPUT_FILE = ""
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/train/NewselaSimpleTest.pbz2")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/train/DebugTrain.pbz2")
INPUT_DATA_FILE = "./data/train/NoOverlapTrain.pbz2"
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5000"))
MODEL_INPUT_FILE = ""
#MODEL_INPUT_FILE = "GRU-2017-03-28-10-21-0-319-100.dat.npz"
def outfilename():
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  return "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
  (voc,vecs,sentences) = pickle.load(handle)

EMBEDDING_DIM = vecs.shape[0]
OUTPUT_DIM = vecs.shape[1]

inputs = np.asarray( [[w for w in sent[:-1]] for sent in sentences])
outputs = np.asarray( [[w for w in sent[1:]] for sent in sentences])

#print inputs[0:5]
#print outputs[0:5]

'''
maxx = 0
for j in range(inputs.shape[0]):
  if len(inputs[j]) > maxx: maxx = len(inputs[j])
print "MAXX %d" % maxx
'''
print "Embedding dim %d" % EMBEDDING_DIM
print "Vocab size %d" % OUTPUT_DIM
print "Output dim %d" % OUTPUT_DIM
print 'shape of inputs', inputs.shape
print 'shape of outputs', outputs.shape
#sys.exit(1)

# Build model
if MODEL_INPUT_FILE == '':
  model = GRUTheano(word_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, bptt_truncate=-1,wordEmbed=vecs)
  save_model_parameters_theano(model, 'INIT-' + outfilename())
else:
  model = load_model_parameters_theano(MODEL_INPUT_FILE,wordEmbed=vecs)


# Print SGD step time
t1 = time.time()
model.sgd_step(inputs[1], outputs[1], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()



# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(inputs[:1000], outputs[:1000])
  #loss = model.calculate_loss(inputs[:], outputs[:])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss: %f" % loss)

  save_model_parameters_theano(model, outfilename())
  print("\n")
  sys.stdout.flush()

train_with_sgd(model, inputs, outputs, learning_rate=LEARNING_RATE, nepoch=NEPOCH, decay=0.9, callback_every=PRINT_EVERY, callback=sgd_callback)

