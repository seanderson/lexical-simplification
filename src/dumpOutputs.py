#! /usr/bin/env python
# 

import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from gru_theano_embed import GRUTheano
import bz2
import cPickle as pickle

EMBED = False

#INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/test/NewselaSimple02test.pbz2")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/train/NewselaSimple03.pbz2")
INPUT_DATA_FILE = "./data/train/DebugRawTrain.pbz2"

# Load data
if EMBED:
  with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
    (voc,vecs,sentences) = pickle.load(handle)
  print "Vocab vector dimensions", vecs.shape
else:
  with bz2.BZ2File(INPUT_DATA_FILE,'r') as handle:
    (voc,sentences) = pickle.load(handle)
    print "Vocab size", len(voc)

print "Num sentences %d" % len(sentences)


#inputs = np.asarray( [[w for w in sent[:-1]] for sent in sentences])
#outputs = np.asarray( [[w for w in sent[1:]] for sent in sentences])

for sent in sentences[:2000]:
  for w in sent:
    print w,voc[w]
