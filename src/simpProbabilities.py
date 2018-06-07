'''
 Combine probabilities of predicted words (nnet or ngram) with word
 changes from aligned sentences.

Error types when matching fails:
;

'''

import sys
#print "path",sys.path
from alignutils import *
import newselautil as nsla
import align
#from prepData import *
#import utils
#import plots
from globdefs import *
import bz2
import pickle
from indexfiles import *

NO_MATCH = -1
RMLIST = (SENT_START_IDX,SENT_END_IDX,PAR_START_IDX)

def createAlignments():
    articles = nsla.loadMetafile()
    nslugs = len(articles)
    lev1 = 0
    lev2 = 1

    num = 0
    for i in range(100): # range(nslugs):
        slug = articles[i]['slug']
        lang = articles[i]['language']
        version= int(articles[i]['version'])
        if lang != 'en' or version != 0: continue
        align.sim_in_articles(slug,0,4,articles)
        num += 1
    print "num",num


def markSimplified(spairs):
    '''
    Mark all words in first list of sentences that are not found in second
    list of sentences.  Ignore case. duplicates, word order.
    Simplified words bracketed by underscores.
    :param spairs list of list of sentence-pair-strings.  Each sentence pair is one alignment.
    '''
    newpairs = [ ]
    for pair in spairs:
        #s1list = [ w for s in pair[0] for w in s.split() ]
        simplified = False
        s2list = set([ w.lower() for w in nsla.tokenize(pair[1]) ])
        '''if len(pair[0]) > 1 or len(pair[1]) > 1:
            print 'p[0]', pair[0]
            print 'p[1]', pair[1]
            print 's1',s1list
            print 's2',s2list
        '''
        FILL = '@IGNORE'
        #new_s1list = [ ]
        sent = pair[0]
        #for sent in pair[0]:
        new_s = [ ]
        for w in nsla.tokenize(sent):
            lw = w.lower()
            if lw == '':
                new_s.append( FILL )
            elif lw not in s2list: # not found
                new_s.append( '_' + w + '_')
                simplified = True
            else:
                new_s.append(w)
        #new_s1list.append( ' '.join(new_s))
        newpairs.append( (' '.join(new_s) , pair[1], simplified) )
    return newpairs



def getParagraphsFiles(versionlst,levellst):
    '''Extract sentences from version list that are 
    within levellst.
    Return files accessed and all paragraphs.'''

    articles = nsla.loadMetafile()
    pars = [ ] # list of articles
    fnames = [ ]
    numpars = [ ] #num paragraphs per file
    # each article is list of paragraphs, each is list of tokenized sent.
    for i,art in enumerate(articles):
        if art['language'] != 'en': continue
        version = int(art['version'])
        readlvl = int(float(art['grade_level']))
        if version not in versionlst or readlvl not in levellst: continue
        fnames.append ( '.'.join( [ art['slug'], "en" , art['version'] ] ) )
        article_pars = nsla.getTokParagraphs(art)
        numpars.append( len(article_pars) )
        for p in article_pars: 
            pars.append(p)
        #if i > 10: break # tmp debug
    return fnames,numpars,pars




def matchword(w,s,i):
    '''Find value of i so that w==s[i].  
    Return this index or NO_MATCHN for failure.'''
    while i < len(s):
        if w == s[i]: return i
        elif s[i]==WORD_NUMERIC and isNumeric(w): return i
        elif s[i]==WORD_UNKNOWN: return i
        i += 1
    return NO_MATCH

def align_words(asent,sent):
    ''' Match each word in asent (list of words) with 
    a word in sent.  Returns list of indices indicating the
    match. A NO_MATCH indicates no match. '''
    # Skip first word (since nnet predicts from second fwd.)
    sj = 1
    # Loop over asent, matching words in asent until all are matched.
    # Must deal with @NUM @UNK
    matches = [NO_MATCH] * len(asent)
    aj = 0
    while aj < len(asent):
        if asent[aj][0] == '_':
            w = asent[aj][1:-1]
        else:
            w = asent[aj]
            # position of exact match -1 if none
        #print "matching",w
        matches[aj] = matchword(w,sent,sj)
        #if matches[aj] > -1: print "    to", matches[aj],sent[matches[aj]]
        if matches[aj] > NO_MATCH:
            sj = matches[aj] + 1
        aj += 1
    #print matches
    return matches

def stripDelimiters(sent):
    '''Return the sent (list of indices) with RMLIST items
    removed.'''
    return [w for w in sent if w not in RMLIST]

def sentenceData(nnetfile,indexfile,probsfile,alignDir):
    '''
    Get all sentences from nnetfile.
    Return all articles in correct order.
    Get probabilities created from network run over nnetfile.
    '''
    probdat = {}
    fprobs = bz2.BZ2File(probsfile,'r')

    # Use training vocabulary for testing data
    with bz2.BZ2File(nnetfile,'r') as handle:
        (invoc,vecs,sentences) = pickle.load(handle)
    sidx = -1
    filenames,npars,sindlst = read_index_file(indexfile)
    articles = nsla.loadMetafile()
    MARKERS = [SENT_START,SENT_END,PAR_START]

    # For each article, get alignment, mark level 0 words.
    # Process probs from each word, verifying words as you go.
    for i,art in enumerate(filenames[:]):
        slug,lang,lvl = art.split('.')
        alignfile = alignDir + slug + "-cmp-0-1.csv"
        pos = sentence_alignments(alignfile) # what is this?
        sentpairs = get_aligned_sentence_strings(articles,slug,0,1)
        print "Article:",slug
        sp  = markSimplified(sentpairs)
        #printPairs(sp)
        for ip in range(npars[i]):  # each par for art

            for js in range(sindlst[i][ip]): # sentences for each par
                sidx += 1
                #print 'sent:',sentences[sidx]
                print 'sent:',"\t".join( [invoc[y] for y in sentences[sidx]][1:]).encode("utf-8")
                sent = [invoc[y] for y in sentences[sidx]] # input sentence
                iarr,arr = utils.readprobs(fprobs,ncol=3) # probs for one sentence
                if iarr == [ ]:
                    print "Error: Ran out of probabilities in sentenceData"
                    sys.exit(-1)

                idx = pos.get( (ip,js) , 0)
                if idx == 0:
                    print "asent NOT Aligned\n--------------"
                    continue # sentence not aligned
                #print 'xx',ip,js,idx
                asent =  sp[idx[0]][0][idx[1] ]  # aligned sent w/ marked words
                print 'asent:',asent.encode('utf-8')

                # check length against probs version
                #nmarkers = len(np.intersect1d(MARKERS, sentences[sidx]))
                #print "LEN:",len(iarr),len( asent.split() ),len(sentences[sidx])

                #matches = align_words(asent.split(),sent)
                #print 'm',matches
                #print "%d %d" % (len(p[0][0].split()),len(iarr))
                # loop over aligned, collecting probs
                sentCleaned = stripDelimiters(iarr[:,-1])
                if len(asent.split()) != len(sentCleaned):
                    print "Error in lengths"
                    print "ELEN1",asent.encode("utf-8")
                    print "ELEN2"," ".join( [invoc[y] for y in sentCleaned]).encode("utf-8")
                    print "ELEN3",sentCleaned
                    continue
                # asent and sentences match
                updateProbs(asent.split(),iarr,arr,probdat)
                outputsentlst = hilightWords(iarr,arr,invoc)
                print 'pred:', "\t".join(outputsentlst).encode("utf-8")
                print 'pred:','\t' # for spacing
                print "--------------"                

    '''for k in sorted(probdat.keys()):
        print "%s %d %f" % (k,len(probdat[k]),np.mean(probdat[k]))
    fig,ax = plots.inithist(8)
    pltnum = 0
    for k in sorted(probdat.keys())[2:]:
        plots.hist(fig,ax,probdat[k],1000,k,8,pltnum)
        pltnum += 1
        print k,
        for x in probdat[k]:
            print x,
        print

    plots.show()
    '''
    fprobs.close()



def updateProbs(asent,iarr,arr,probs):
    ''' 
    For each matching word in asent, keep probs ratio, separating
    correct from incorrect.  Update probdat, a dictionary of lists, each
    list containing probabilities for that type.
    '''
    CORRECT = ["correct","incorrect"]
    SIMPLE = ["simple","complex"]
    KNOWN = ["known","unknown"]
    sentCleaned = stripDelimiters(iarr[:,-1])  
    if len(asent) != len(sentCleaned):
        print "Error asentXX",asent," scleaned",sentCleaned
        sys.exit(-1)
    probdat = utils.wordStats(arr,iarr)
    idx = 0 # index into asent
    for i in range( len(iarr) ):
        output_word = iarr[i][-1]
        if output_word in RMLIST: continue

        c,s,k = "","",""
        if probdat[i][0] == 1: c = "correct"
        else: c = "incorrect"
        if asent[idx][0] != '_': s = "simple"
        else: s = "complex"
        if output_word == UNK_IDX: k = "known"
        else: k = "unknown"
        key = "-".join([c,s,k])
        if probs.get(key,None) == None: probs[key] = [ ]
        probs[key].append(probdat[i][-1])
        if probs.get("actual-"+c,None) == None:
            probs["actual-"+c] = [ ]
        probs["actual-"+c].append(probdat[i][-1])
        idx += 1

def hilightWords(iarr,arr,invoc):        
    #sentCleaned = stripDelimiters(iarr[:,-1])
    probdat = utils.wordStats(arr,iarr)
    sent = [ ]
    for i in range( len(iarr) ):
        word = invoc[probdat[i][2]]
        if probdat[i][-1] > 0.1:  # strong prob
            sent.append( "_"+word)
        else:
            sent.append(word)
    return sent
            
def printPairs(sp):
    for p in sp:
        if p[2]:
            print p[0]
            print p[1]
            print '----'
    
def findSimplifiedWords():
    '''
    Test ability to align sentences and mark words in article0 that
    are not in article1.
    '''
    articles = nsla.loadMetafile()
    for i in range(10): # range(nslugs):
        slug = articles[i]['slug']
        lang = articles[i]['language']
        version= int(articles[i]['version'])
        if lang != 'en' or version != 0: continue
        print "Article: ", slug

        sentpairs = get_aligned_sentence_strings(articles,slug,0,1)
        #for s in sentpairs: print s
        sp  = markSimplified(sentpairs)
        #for s in sp: print s[0], "\n"   , s[1], "\n"

# allProbs has all sentences
# alignment
BASEDIR = "/Users/sven/res/locsimp/wpred/"
nnetFile =  BASEDIR + "/data/test/NoOverlapTest.pbz2"
indexFile = BASEDIR + "/data/test/NoOverlapTest.idx"
probsFile = BASEDIR + "/100hu/res-3/allprobs.pbz2"
alignDir = "/home/sven/res/alignment/output/sentences/"
#slugs = sentenceData(nnetFile,indexFile,probsFile,alignDir)
#align.all()
findSimplifiedWords()
