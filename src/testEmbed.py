#! /usr/bin/env python
# Run save GRU model on testing data.

import sys
import os
import time
import numpy as np
from utils_embed import *
from datetime import datetime
from gru_theano_embed import GRUTheano
import bz2
import cPickle as pickle

#INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/train/NewselaSimple02.pbz2")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/test/NoOverlapTest.pbz2")
MODEL_INPUT= sys.argv[1]


def allWordStats(fname,outputs,outfile):
  '''Read all output data and gather stats.'''
  fout = open(outfile,'w')
  fd = bz2.BZ2File(fname,'r')
  for output in outputs:
    iarr,arr = readprobs(fd) # data for one sentence
    if arr == []: break
    wordStats(arr,iarr,output,fout)
  fd.close()
  fout.close()
                    
# Load data
with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
  (voc,vecs,sentences) = pickle.load(handle)
print "Vocab vector dimensions", vecs.shape
print "Num sentences %d" % len(sentences)

EMBEDDING_DIM = vecs.shape[0]
OUTPUT_DIM = vecs.shape[1]

inputs = np.asarray( [[w for w in sent[:-1]] for sent in sentences])
outputs = np.asarray( [[w for w in sent[1:]] for sent in sentences])
#print 'in',inputs[0:5]
#print 'out',outputs[0:5]

# Load model
model = load_model_parameters_theano(path=MODEL_INPUT,wordEmbed=vecs)
print "Loaded model: ",MODEL_INPUT

#loss = model.calculate_loss(inputs,outputs)
#print("Loss: %f" % loss)
#generate_sentences(model,100,index_to_word,word_to_index)

# Gather most probable words and their probabilities
fname,logppl,numwords = allWordProbs(model,inputs[:],outputs[:])
ppl = pow(2.0,(-logppl/float(numwords)))
print "Perplexity %f %d %f" % (logppl,numwords, ppl)
# print stats
tstamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
outfile = "word-%s.stat" % (tstamp)
allWordStats(fname,outputs[:10],outfile=outfile)
                    
'''
(tp,tn,fp,fn) = test_all(model,x_train,y_train)
print "tp %d tn %d fp %d fn %d" % (tp,tn,fp,fn)
precision = float(tp)/(tp + fp)
recall = float(tp)/(tp+fn)
accuracy = float(tp+tn)/(tp+fp+tn+fn)
fscore = 2.0 * (precision * recall) / (precision + recall)
gscore = 2.0 * (accuracy * recall) / (accuracy + recall)
print "G/F/P/R/A %5.3f %5.3f %5.3f %5.3f %5.3f" % (gscore,fscore,precision,recall,accuracy)

save_output(model,x_train,y_train)
'''
