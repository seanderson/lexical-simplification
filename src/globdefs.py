'''
  globdefs.py:
  Global definitions used by numerous scripts.
'''

import nltk
from nltk.corpus import stopwords
STOPWORDS = sorted(stopwords.words('english'))

SPC = " "
#special "words" added to vocabulary
# words not in word2vec lexicon rep. with special vectors
WORD_UNKNOWN = '@UNK'
WORD_NUMERIC = '@NUM' # unknown numeric word
PAR_START = "@PAR"
SENT_START = "@SENT"
SENT_END = "@SEND"
W_PERIOD = '.'
W_COMMA = ','
W_QUESTION = '?'
W_EXCLAMATION = '!'
W_SEMICOLON = ';'
W_COLON = ':'
SPECIAL_WORDS = [WORD_UNKNOWN,SENT_START,SENT_END,PAR_START,WORD_NUMERIC,W_PERIOD,W_COMMA,W_QUESTION,W_EXCLAMATION,W_SEMICOLON,W_COLON] + STOPWORDS
NUM_SPECIAL_WORDS = len(SPECIAL_WORDS)
UNK_IDX = 0

SENT_START_IDX = 1
SENT_END_IDX = 2
PAR_START_IDX = 3



STANFORD_POS_BASE="/home/sven/lib/taggers/stanford-postagger-full-2014-08-27/"
STANFORD_MODEL=STANFORD_POS_BASE + "models/english-bidirectional-distsim.tagger"
STANFORD_JAR=STANFORD_POS_BASE+ "stanford-postagger.jar"
