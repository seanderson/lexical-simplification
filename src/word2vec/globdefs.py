'''
  globdefs.py:
  Global definitions used by numerous scripts.
'''
import sys

sys.path.append('../')
from nltk.corpus import stopwords

STOPWORDS = sorted(stopwords.words('english'))

SPC = " "
# special "words" added to vocabulary
# words not in word2vec lexicon rep. with special vectors
WORD_UNKNOWN = '@UNK'
WORD_NUMERIC = '@NUM'  # unknown numeric word
PAR_START = "@PAR"
SENT_START = "@SENT"
SENT_END = "@SEND"
W_PERIOD = '.'
W_COMMA = ','
W_QUESTION = '?'
W_EXCLAMATION = '!'
W_SEMICOLON = ';'
W_COLON = ':'
SPECIAL_WORDS = [WORD_UNKNOWN, SENT_START, SENT_END, PAR_START, WORD_NUMERIC,
                 W_PERIOD, W_COMMA, W_QUESTION, W_EXCLAMATION, W_SEMICOLON,
                 W_COLON] + STOPWORDS
NUM_SPECIAL_WORDS = len(SPECIAL_WORDS)
UNK_IDX = 0

SENT_START_IDX = 1
SENT_END_IDX = 2
PAR_START_IDX = 3

STANFORD_POS_BASE = "/home/sven/lib/taggers/stanford-postagger-full-2014-08-27/"
STANFORD_MODEL = STANFORD_POS_BASE + "models/english-bidirectional-distsim.tagger"
STANFORD_JAR = STANFORD_POS_BASE + "stanford-postagger.jar"

DATA_METAFILE = "/home/nlp/wpred/text_databases/metafile.txt"
VOCFILE = "/home/nlp/wpred/text_databases/voc.tsv"
VOC_HDR_WORD = "Word"
VOC_HDR_TOTAL = "Total"

# PENN tagset
# ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS'])

# Major word classes are reduced to single forms (see Paetzold, 2016)


def getGeneralisedPOS(tag):
    """
    This function is from lexenstein
    Returns a generalised version of a POS tag in Treebank format.
    @param tag: POS tag in Treebank format.
    @return A generalised POS tag.
    """
    if tag.startswith('N'):
        result = 'N'
    elif tag.startswith('V'):
        result = 'V'
    elif tag.startswith('RB'):
        result = 'A'
    elif tag.startswith('J'):
        result = 'J'
    elif tag.startswith('W'):
        result = 'W'
    elif tag.startswith('PRP'):
        result = 'P'
    else:
        result = tag.strip()
    return result


UPENN_TAGSET_FILE = 'help/tagsets/upenn_tagset.pickle'
POS_DELIM = '_'
MODEL_FILE = 'w2v.mdl'
