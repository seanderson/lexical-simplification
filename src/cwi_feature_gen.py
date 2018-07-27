'''
Baseline CWI classifier for BEA18 data.
Creates following features from Yimam (2017) Complex Word Identification...
'''

from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
#from nltk.tokenize import TreebankWordTokenizer
#Wordtokenizer = TreebankWordTokenizer()
import nltk.data
Tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
morph = MorphAdornerToolkit('./MorphAdornerToolkit/')

dataDir = 'english/'
infile = 'Wikipedia_Dev.tsv'

'''#### Training Data

The training data will be provided in the following format:

    3P7RGTLO6EE07HLUVDKKHS6O7CCKA5	Both China and the Philippines flexed their muscles on Wednesday.	31	51	flexed their muscles	10	10	3	2	1	0.25
    3P7RGTLO6EE07HLUVDKKHS6O7CCKA5	Both China and the Philippines flexed their muscles on Wednesday.	31	37	flexed	10	10	2	6	1	0.4
    3P7RGTLO6EE07HLUVDKKHS6O7CCKA5	Both China and the Philippines flexed their muscles on Wednesday.	44	51	muscles	10	10	0	0	0	0.0

Each line represents a sentence with one complex word annotation and
relevant information, each separated by a TAB character.  - The first
column shows the HIT ID of the sentence. All sentences with the same
ID belong to the same HIT.  - The second column shows the actual
sentence where there exists a complex phrase annotation.  - The third
and fourth columns display the start and end offsets of the target
word in this sentence.  - The fifth column represents the target word.
- The sixth and seventh columns show the number of native annotators
and the number of non-native annotators who saw the sentence.  - The
eighth and ninth columns show the number of native annotators and the
number of non-native annotators who marked the target word as
difficult.  - The tenth and eleventh columns show the gold-standard
label for the binary and probabilistic classification tasks.

The labels in the binary classification task were assigned in the following manner:
- 0: simple word (none of the annotators marked the word as difficult)
- 1: complex word (at least one annotator marked the word as difficult)

The labels in the probabilistic classification task were assigned as
`<the number of annotators who marked the word as difficult>`/`<the
total number of annotators>`.

'''
# Convert field name to its position
FIELDS = ['hit','sent','start','stop','target','num-l1','num-l2','complex-l1','complex-l2','lbl-bin','lbl-prob']
INT_FIELDS = ['start','stop','num-l1','num-l2','complex-l1','complex-l2','lbl-bin']
FLT_FIELDS = ['lbl-prob']
fileidx = dict( [(w,i) for i,w in enumerate(FIELDS)] )
#print fieldidx


def tokenizeSent(item):
    '''Tokenize sentence.  The target may be several words, so it must be
    treated as one string.
    '''
    target = item[ fileidx['target'] ]
    begin = item[ fileidx['start']]
    end = item[ fileidx['stop']]
    s = item[ fileidx['sent'] ]
    lst = [ ]
    if begin > 0:
        lst = s[0:begin].split()
        lst.append(target)
    else:
        lst.append(target)
    if end < len(s):
        lst = lst + s[end:].split()
    return lst

def loadData(fname):
    '''Load, convert, and tokenize input data: one item per line.
    '''
    dat = [ ]
    with open(fname,'r') as ifd:
        for line in ifd.readlines():
            col = line.rstrip().split('\t')
            for x in INT_FIELDS:
               col[ fileidx[x] ] = int( col[ fileidx[x] ] )
            for x in FLT_FIELDS:
               col[ fileidx[x] ] = float( col[ fileidx[x] ] )
            col[ fileidx['sent'] ] = tokenizeSent(col)

            dat.append(col)
    return dat


dat = loadData(dataDir+infile)

'''
#print dat[0:2]
fe = FeatureEstimator(norm=False)
#fe.addLexiconFeature('lexicon.txt', 'Simplicity')
fe.addLengthFeature('Complexity')
fe.addSynonymCountFeature('Simplicity')
fe.addSyllableFeature(morph, 'Complexity')


feats = fe.calculateFeatures('test.cwictor', format='cwictor')
print feats
'''

VOWELS = ['a','e','i','o','u','y','A','E','I','O','U','Y']
VOWEL_REGEX = re.compile( r'[AEIOUYaeiouy]')

def numVowels(s):
    '''Number of vowels in string'''
    return len( re.findall(VOWEL_REGEX,s) )

#w = ' a\t 7 & y o ujkeee\nie'
#print "Num vowels in %s %s" % (w,numVowels(w))


def hitFrequency(item,alldata):
    '''Return (word,count,hitlen) of word for all matching hits in item, a
    row of data for one target word.
    '''
    word = item[fileidx['target']]

    sidx = fileidx['sent']
    hidx = fileidx['hit']
    hit = item[hidx]
    wcount = 0
    hitlen = 0
    for dat in alldata:
        if dat[hidx] == hit:
            hitlen += len(dat[sidx])
            #print dat[sidx]
            wcount += dat[sidx].count(word)
    return (word,wcount,hitlen)

# test hitFreq
for x in dat[:40]:
    #print x
    print hitFrequency(x,dat)
