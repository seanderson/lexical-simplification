# Prepare newsela sentences at a single level for input into an RNN
# This version assumes words are represented by one-hot vectors,
# therefore bypassing most details of vector construction.

from globdefs import *
import newselautil as ns
import cPickle as pickle
import bz2
import re
import numpy as np
import nltk
#import operator
import theano
import sys
import classpaths as paths

# seek digit in string
RE_HASDIGIT = re.compile('\d')
RE_HASCHAR = re.compile('[a-zA-Z]')

IS_NUMBER = 1 # a plain number, expressable as float
IS_NOUN_NUMBER = 2 # a number ending in s
IS_NOT_NUMBER = 0 # neither of the above


def isNumeric(a):
    '''
    return IS_NUMBER/IS_NOUN_NUMBER/IS_NOT_NUMBER
    '''
    try: # Could be a float
        f = float(a)
        return IS_NUMBER
    except:
        pass
    if not re.search(RE_HASDIGIT,a): # has no digit
        return IS_NOT_NUMBER
    # Might be plural number like "90s"
    if a[-1] == 's':
        try:
            f = float(a[:-1])
            return IS_NOUN_NUMBER
        except:
            return IS_NOT_NUMBER
    return IS_NOT_NUMBER

def articleSubset(fname):
    files = [ ]
    with open(fname,'r') as infile:
        for line in infile.readlines():
            name,lang,level,garb = line.split('.')
            files.append(name)
    return files


def getParagraphs(versionlst,levellst,art_subset=""):
    '''Extract sentences from version list that are 
    within levellst.  Uses subset of articles if arg specified.'''
    artlist = [ ]
    if art_subset!="":
        artlist = articleSubset(art_subset)
        
    articles = ns.loadMetafile()
    pars = [ ] # list of articles
    # each article is list of paragraphs, each is list of tokenized sent.
    for i,art in enumerate(articles):
        # if subset selected, include only articles in the subset
        if artlist != [ ] and art['slug'] not in artlist: continue
        if art['language'] != 'en': continue
        version = int(art['version'])
        readlvl = int(float(art['grade_level']))
        if version not in versionlst or readlvl not in levellst: continue
        for p in ns.getTokParagraphs(art):
            pars.append(p)
    return pars

def getParagraphsFiles(versionlst,levellst,art_subset=""):
    '''Extract sentences from version list that are 
    within levellst.
    Return files accessed, in order, and all paragraphs.'''
    artlist = [ ]
    if art_subset!="":
        artlist = articleSubset(art_subset)
    articles = ns.loadMetafile()
    pars = [ ] # list of articles
    fnames = [ ]
    numpars = [ ] #num paragraphs per file
    # each article is list of paragraphs, each is list of tokenized sent.
    for i,art in enumerate(articles):
        # if subset selected, include only articles in the subset
        if artlist != [ ] and art['slug'] not in artlist: continue
        if art['language'] != 'en': continue
        version = int(art['version'])
        readlvl = int(float(art['grade_level']))
        if version not in versionlst or readlvl not in levellst: continue
        fnames.append ( '.'.join( [ art['slug'], "en" , art['version'] ] ) )
        article_pars = ns.getTokParagraphs(art)
        numpars.append( len(article_pars) )
        for p in article_pars: 
            pars.append(p)
        #if i > 10: break # tmp debug
    return fnames,numpars,pars

def countLevels():
    '''Get count of grade levels for articles-LVL.'''
    articles = ns.loadMetafile()
    for level in range(0,5):
        versionLevels = {}
        for k in range(2,13):
            versionLevels[k] = 0
        i = 0
        while i < len(articles):
            version = articles[i]['version']
            readlvl = articles[i]['grade_level']
            slug = articles[i]['slug']
            fname = articles[i]['filename']
            lang = articles[i]['language']
            if lang == 'en' and int(version) == level:
                rlev = int(float(readlvl))
                versionLevels[rlev] += 1
            i += 1

        print level,
        for k in range(2,13):
            print "%d" % (versionLevels[k]),
        print

def build_w2v_lexicon(outfile="lex.pbz2",vocabsize=5000,art_subset=""):
    '''Create and store lexicon of all word2vec
    words found in the newsela articles levels 1 to 4 (train).'''

    pars = getParagraphs( [1,2,3,4], list(range(0,13)) , art_subset )
    lex = list()
    # pars is list of paragraphs
    # each par is list of strings (sentences).
    for p in pars:
        for s in p:
            words = ns.tokenize(s)
            for w in words:
                lex.append(w)
    word_freq = nltk.FreqDist(lex)
    print "Lex size %d" % (len(lex))
    print("Found %d unique word tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    #sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    #for v in vocab:
    #    print "%d\t%s" % (v[1],v[0].encode('utf-8'))
    #sys.exit(1)
    # Add words, in order of frequency, that occur in
    # word2vec vocab.
    import gensim
    CORPUS_DIR = '/home/sven/res/corpora/word2vec/'
    model = gensim.models.Word2Vec.load_word2vec_format(CORPUS_DIR+'GoogleNews-vectors-negative300.bin', binary=True)
    vecs = { }
    i = 0
    while i < len(vocab) and len(vecs) < vocabsize:
        w = vocab[i][0]
        try: # seek word w
            vec = model[w]
            vecs[w] = vec
        except KeyError: # seek lower case version
            try:
                vec = model[w.lower()]
                vecs[w.lower()] = vec
            except KeyError:
                vecs[w] = '@UNK' # unknown word
        i += 1
    print 'Num in w2vec %d' % (len(vecs.keys()))
    with bz2.BZ2File(outfile,'w') as fd:
        pickle.dump(vecs,fd)

def build_lexicon(outfile="lexNovec.pbz2",vocabsize=5000,art_subset=""):
    '''Create and store lexicon of vocabsize words in the newsela articles
    levels 1 to 4 (train).'''

    pars = getParagraphs( [1,2,3,4], list(range(0,13)) , art_subset )
    lex = list()
    # pars is list of paragraphs
    # each par is list of strings (sentences).
    for p in pars:
        for s in p:
            words = ns.tokenize(s)
            for w in words:
                lex.append(w)
    word_freq = nltk.FreqDist(lex)
    print "Lex size %d" % (len(lex))
    print("Found %d unique word tokens." % len(word_freq.items()))
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    with bz2.BZ2File(outfile,'w') as fd:
        pickle.dump(vocab,fd)


def create_sentences(pars,word_to_index, tokenize=True):
    '''Return list of all sentences, mapping words to vector index and
    dealing with sentencestart/unknown/numeric words.  Also returns
    list of [numsentences], one for each par.

    '''
    all_sents = [ ]
    index = [ ]
    for ip,p in enumerate(pars):
        line = [ word_to_index[PAR_START] ]
        index.append( len(p) )
        for s in p:
            line.append(word_to_index[SENT_START])
            if tokenize:
                words = ns.tokenize(s)  # list of words
            else:
                words = s.split(' ')
            for w in words:
                idx = word_to_index.get(w,UNK_IDX)
                if idx == UNK_IDX: # try lower case match
                    idx = word_to_index.get(w.lower(),UNK_IDX)
                if idx == UNK_IDX and isNumeric(w) != IS_NOT_NUMBER: # numeric
                    idx = word_to_index[WORD_NUMERIC]
                line.append( idx )
            line.append(word_to_index[SENT_END])
            all_sents.append(line)
            line = [ ]
    return all_sents,index
    
def write_index_file(OUTINDEX,filenames,numpars,sindexlst):
    '''Writes file of articles/pars/sentences in format:
    #articles
    artname  numpars numS_par0 numS_par1 ...
    '''
    sidx = 0
    with open(OUTINDEX,'w') as findex:
        findex.write("%d\n" % len(filenames))
        for i,f in enumerate(filenames):
            findex.write("%s %d" % (f,numpars[i]))
            for j in range(numpars[i]):
                findex.write(" %d" % sindexlst[sidx])
                sidx += 1
            findex.write("\n")

def read_index_file(INDEXFILE):
    '''Reads index file.'''
    fnames = [ ]
    numpars = [ ]
    sindexlst = [ ]
    with open(INDEXFILE,'r') as findex:
        n = int(findex.readline().rstrip('\n'))

        for i in range(n):
            dat = findex.readline().rstrip('\n').split()
            fnames.append( dat[0] )
            numpars.append(int(dat[1]))
            sindexlst.append( map(int,dat[2:]))
    return fnames,numpars,sindexlst

def createTestData(intrainfile,outtestfile,idxfile,artSubset=''):
    '''Create training data for newsela articles using
    word2vec embeddings for words.
    '''
    # Use training vocabulary for testing data
    with bz2.BZ2File(intrainfile,'r') as handle:
        (invoc,sentences) = pickle.load(handle)
    sentences = None

    #sys.exit(1)
    # Map from key position to vector
    word_to_index = dict([(w, i) for i, w in enumerate(invoc)])
    
    filenames,numpars,pars = getParagraphsFiles( [0], [12], artSubset ) # Grade 12, Level 0

    all_sents,sindexlst = create_sentences(pars,word_to_index)
    write_index_file(idxfile,filenames,numpars,sindexlst)
            
    with bz2.BZ2File(outtestfile,'w') as fd:
        pickle.dump( (invoc,all_sents) , fd)



def createValidationData(intrainfile,outtestfile,idxfile,artSubset=''):
    '''Create training data for newsela articles using
    word2vec embeddings for words.
    '''
    # Use training vocabulary for testing data
    with bz2.BZ2File(intrainfile,'r') as handle:
        (invoc,sentences) = pickle.load(handle)
    sentences = None

    # Map from key position to vector
    word_to_index = dict([(w, i) for i, w in enumerate(invoc)])

    filenames,numpars,pars = getParagraphsFiles( [1,2,3,4], list(range(2,10)), artSubset ) # test subset, excluding Grade 12 and Level 0

    all_sents,sindexlst = create_sentences(pars,word_to_index)
    write_index_file(idxfile,filenames,numpars,sindexlst)
            
    with bz2.BZ2File(outtestfile,'w') as fd:
        pickle.dump( (invoc,all_sents) , fd)


        
def add_special_wordvecs(wordvecs,keys,invoc,outvoc):
    '''
    keys all words in w2v vocab
    invoc/outvoc all keys in/out both newsela and w2v vocab.
    Return invoc,vecs ith item invoc is ith vector.
    Add all special words to front of keys.
    All special words not in invoc have unique one-hot vectors added.
    They are at beginning of returned vecs.
    '''

    invoc = special_subset + invoc
    num_special = len(special_subset)
    vocabsize = len(invoc)
    veclen = len(wordvecs['the']) + num_special
    vecs = np.zeros( (veclen,vocabsize) , dtype=theano.config.floatX)

    # vector codes for special words not in vocabulary
    for i in range(num_special):
        vecs[i][i] = 1.0
    for i in range(num_special,vocabsize):
        vecs[num_special:,i] = wordvecs[invoc[i]]
    return invoc,vecs

def createTrainData(vocfile='data/LexiconOneHotAll.pbz2', outfile = 'NewselaSimpleXX.pbz2',artSubset='',lexsize=0):
    '''Create training data for newsela articles using
    one-hot word vectors.
    '''

    with bz2.BZ2File(vocfile,'r') as infile:
        lex = pickle.load(infile)
    print "Loaded lexicon %d words" % len(lex)
    voc = map( lambda x: x[0], lex[:lexsize])
    # Create special vectors for special words NOT in invoc
    special_subset = [ ]
    for w in SPECIAL_WORDS[::-1]: # critical to get UNK in position zero
        if w not in voc:
            voc.insert(0,w)
    print "Final vocab size: %d" % len(voc)

    # Map from key position to vector
    word_to_index = dict([(w, i) for i, w in enumerate(voc)])
    pars = getParagraphs( [1,2,3,4], list(range(2,10)),artSubset )
    all_sents,sentenceindex = create_sentences(pars,word_to_index)

    with bz2.BZ2File(outfile,'w') as fd:
        pickle.dump( (voc,all_sents) , fd)

def testTrain():
    with bz2.BZ2File('NewselaSimple01.pbz2','r') as fd:
        (voc,vecs,sents) = pickle.load(fd)

    for s in sents[:100]:
        for x in s:
            print voc[x],
        print 

def printSentenceData(filename):

    with bz2.BZ2File(filename,'r') as handle:
        (invoc,vecs,sentences) = pickle.load(handle)
    for sidx in range(len(sentences)):
        sent = [invoc[y] for y in sentences[sidx]] # input sentence
        print "|".join(sent)

        
def main():
    #getParagraphs( [1,2,3,4], list(range(2,10)) )
    # 21105 have 10 or more tokens, 20754 with hypen splitting
    #getParagraphs( [1,2,3,4], list(range(2,10)), 'data/trainArticles.txt' )
    # 19560 have 10 or more, restricted to the training list
    #build_w2v_lexicon(19560,'data/trainArticles.txt')

    #with bz2.BZ2File('lex.pbz2','r') as fd:
    #    vecs = pickle.load(fd)
    #for w in vecs.keys():
    #    if vecs[w] == '@UNK':
    #        print w.encode('utf-8')
    

    buildType = "NoOverlapRaw" # "DebugRaw"
    if buildType == "DebugRaw":
        lexsize = "1000"
        trainSubset = paths.BASEDIR + '/data/trainArticleSubset.txt'
        testSubset = paths.BASEDIR + '/data/testArticleSubset.txt'
        vocFile = paths.BASEDIR + '/data/LexiconOneHot'+lexsize+'.pbz2'
        trainFile = paths.BASEDIR + '/data/train/'+buildType+'Train.pbz2'
        testFile = paths.BASEDIR + '/data/test/'+buildType+'Test.pbz2'
        validFile = paths.BASEDIR + '/data/train/'+buildType+'Valid.pbz2'
        idxFile = paths.BASEDIR + '/data/test/'+buildType+'Test.pbz2'
        idxFile = paths.BASEDIR + '/data/test/DebugTest.idx'
        idxValidFile = paths.BASEDIR + '/data/train/DebugValid.idx'
    elif buildType == "NoOverlapRaw":  # no word2vec embedding
        lexsize = "19560"
        #lexsize = "1000"
        trainSubset = paths.BASEDIR + '/data/trainArticles.txt'
        testSubset = paths.BASEDIR + '/data/testArticles.txt'
        vocFile = paths.BASEDIR + '/data/LexiconOneHotAll.pbz2'
        trainFile = paths.BASEDIR + '/data/train/'+buildType+'Train.pbz2'
        testFile = paths.BASEDIR + '/data/test/'+buildType+'Test.pbz2'
        validFile = paths.BASEDIR + '/data/train/'+buildType+'Valid.pbz2'
        idxFile = paths.BASEDIR + '/data/test/'+buildType+'Test.idx'
        idxValidFile = paths.BASEDIR + '/data/train/'+buildType+'Valid.idx'
    else:
        sys.exit(1)

    # One Hot vocab vectors
    # build_lexicon(vocFile,int(lexsize),trainSubset)
    
    createTrainData(vocFile,trainFile,trainSubset,int(lexsize))
    createTestData(trainFile,testFile,idxFile,testSubset)
    createValidationData(trainFile,validFile,idxValidFile,testSubset) # validatio
    #filenames,numpars,sindexlst = read_index_file("./data/test/NewselaSimple03test.idx")
    #write_index_file("./foo.idx",filenames,numpars,sindexlst)
    #createTestData()
    #testTrain()
    
def printPars():
    pars = getParagraphs( [0], [12], 'data/testArticleSubset.txt')
    lex = list()
    for p in pars[:100]:
        for s in p:
            words = ns.tokenize(s)
            for w in words:
                #lex.append(w)
                print w,'|',
            print

if __name__ == "__main__":
    main()
    #printSentenceData('data/test/DebugTest.pbz2')
    #printSentenceData('data/test/NoOverlapTest.pbz2')
    #printPars()
    
    # Use training vocabulary for testing data
    #nnetfile = "./data/test/NoOverlapTest.pbz2"
    #with bz2.BZ2File(nnetfile,'r') as handle:
    #    (invoc,vecs,sentences) = pickle.load(handle)
    #for s in sentences:
    #    print 'sent:',"\t".join( [invoc[y] for y in s][1:] )
