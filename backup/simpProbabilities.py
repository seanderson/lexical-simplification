"""
Combine probabilities of predicted words (nnet or ngram) with word
changes from aligned sentences.
"""

from alignutils import *
import newselautil as nsla
import align
from prepData import *
import utilsKeras as utils
import h5py
import pylab as pl

RMLIST = (SENT_START_IDX, SENT_END_IDX, PAR_START_IDX)  # the list used in stripDelimiters


def createAlignments():
    nslugs = 100  # -1 = everything
    levels = [(0, 4, 2)]
    align.align_first_n(nslugs, levels)


def getParagraphsFiles(versionlst, levellst):
    """Extract sentences from version list that are
    within levellst.
    Return files accessed and all paragraphs."""

    articles = nsla.loadMetafile()
    pars = []  # list of articles
    fnames = []
    numpars = []  # num paragraphs per file
    # each article is list of paragraphs, each is list of tokenized sent.
    for i, art in enumerate(articles):
        if art['language'] != 'en':
            continue
        version = int(art['version'])
        readlvl = int(float(art['grade_level']))
        if version not in versionlst or readlvl not in levellst:
            continue
        fnames.append('.'.join([art['slug'], "en", art['version']]))
        article_pars = nsla.getTokParagraphs(art)
        numpars.append(len(article_pars))
        for p in article_pars:
            pars.append(p)
        # 1f i > 10: break  # tmp debug
    return fnames, numpars, pars


def matchword(w, s, i):
    """
    Find value of i so that w==s[i].
    Return this index or NO_MATCHN for failure.
    :param w: a word (string)
    :param s: a list of words
    :param i: the index to start the search from
    """
    while i < len(s):
        if w == s[i] or isNumeric(w) or s[i] in [WORD_NUMERIC, WORD_UNKNOWN]:
            return i
        i += 1
    return NO_MATCH


def align_words(asent, sent):
    """
    Match each word in asent (list of words) with
    a word in sent.  Returns list of indices indicating the
    match. A NO_MATCH indicates no match.
    :param asent: a string from the aignment output (simplified words are
                  marked with underscored)
    :param sent: the sentence from the nn output
     """
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
        # print("matching",w)
        matches[aj] = matchword(w, sent, sj)
        # if matches[aj] > -1: print("    to", matches[aj],sent[matches[aj]])
        if matches[aj] > NO_MATCH:
            sj = matches[aj] + 1
        aj += 1
    if NO_MATCH in matches:
        print("the sentences cannot be matched:\n"+ str(matches))
    return matches


def stripDelimiters(sent):
    """Return the sent (list of indices) with RMLIST items
    removed."""
    return [w for w in sent if w not in RMLIST]


def sentenceData(nnetfile, indexfile, probsfile):
    """
    Get all sentences from nnetfile.
    Return all articles in correct order.
    Get probabilities created from network run over nnetfile.
    :param indexfile: contains filename then number of paragraphs then num sentences per para
    """
    h5fd = h5py.File(probsfile, 'r')

    # Use training vocabulary for testing data
    with bz2.BZ2File(nnetfile,'r') as handle:
        # (invoc,sentences) = pickle.load(handle)
        (invoc, vecs, sentences) = pickle.load(handle)
    sidx = -1
    filenames,npars,sindlst = read_index_file(indexfile)
    articles = nsla.loadMetafile()
    MARKERS = [SENT_START,SENT_END,PAR_START]
    # For each article, get alignment, mark level 0 words.
    # Process probs from each word, verifying words as you go.
    simple = []
    complex = []
    predictions = utils.readProbs(h5fd, invoc)
    for i,art in enumerate(filenames[:]):
        slug,lang,lvl = art.split('.')
        sp = get_aligned_sentences(articles,slug,0,1)
        # print slug
        current_alignment = 0
        curr = -1  # id of the sentence in the current article
        for ip in range(npars[i]):  # each par for art

            for js in range(sindlst[i][ip]): # sentences for each par
                sidx += 1
                curr += 1
                # print 'sent:',sentences[sidx]
                # print 'sent:'," ".join( [invoc[y] for y in sentences[sidx]]).encode("utf-8")
                sent_begin_with_par = [invoc[y] for y in sentences[sidx]][0] == PAR_START  # input sentence
                if len(predictions) <= sidx:
                    print "Error: Ran out of probabilities in sentenceData"
                    return simple, complex
                nn_output = predictions[sidx]
                # idx = pos.get( (ip,js) , 0)
                # if idx == 0: continue # sentence not aligned
                # print 'xx',ip,js,idx
                nn_representation_of_sentence = [invoc[int(x[0][1] - 1)] for x in nn_output]
                # print(' '.join(sent))
                # print(' '.join(nn_representation_of_sentence))
                if current_alignment >= len(sp) or sp[current_alignment].ind0 != curr:
                    # print("alignment not found")
                    continue  # sentence not aligned
                aligned_output = sp[current_alignment]
                # print(aligned_output.sent0)
                # print('')
                aligned_len = len(aligned_output.sent0.split(" "))
                nn_len = len(nn_representation_of_sentence)
                if (sent_begin_with_par and nn_len != aligned_len + 2) or (not sent_begin_with_par and nn_len != aligned_len + 1):
                    print "Error: Lengths are not equal"
                    print(' '.join(nn_representation_of_sentence))
                    print(aligned_output.sent0)
                    continue
                    # sys.exit(-1)
                simple_tmp, complex_tmp = analyze(aligned_output.sent0.split(' '), nn_output,
                                                  nn_representation_of_sentence)
                simple += simple_tmp
                complex += complex_tmp
                """asent =  sp[idx[0]][0][idx[1] ]  # aligned sent w/ marked words
                print 'asent',asent.encode('utf-8')
                # check length against probs version
                #nmarkers = len(np.intersect1d(MARKERS, sentences[sidx]))
                #print "LEN:",len(iarr),len( asent.split() ),len(sentences[sidx])
                print "--------------"
                #matches = align_words(asent.split(),sent)
                #print 'm',matches
                #print "%d %d" % (len(p[0][0].split()),len(iarr))
                # loop over aligned, collecting probs
                sentCleaned = stripDelimiters(sentences[sidx])
                if len(asent.split()) != len(sentCleaned):
                    print "Error in lengths",asent.encode("utf-8"),sentCleaned
                    #sys.exit(-1)
                if len(iarr) != len(sentences[sidx])-1:
                    print "iarrXX",len(iarr),iarr,"\nXsent",len(sent)-1,sen
                if True: continue
                corr,incorr,xcorr,xincorr = updateProbs(asent.split(),sentences[sidx][1:],matches,iarr,arr)
                incorrprobs += incorr
                corrprobs += corr
                xincorrprobs += xincorr
                xcorrprobs += xcorr"""
                current_alignment += 1
    return simple, complex
                
    # print "#corr %d %f incorr %d %f" % (len(corrprobs),np.mean(corrprobs),len(incorrprobs),np.mean(incorrprobs))
    # print "#xcorr %d %f xincorr %d %f" % (len(xcorrprobs),np.mean(xcorrprobs),len(xincorrprobs),np.mean(xincorrprobs))
    # fprobs.close()


def get_statistics():
    simple, complex = sentenceData(nnetFile, indexFile, probsFile)
    pl.hist([x * 1000 for x in simple], bins=range(0, 1001, 1))
    pl.hist([x * 1000 for x in complex], bins=range(0, 1001, 1))
    pl.show()
    print(simple)
    print(complex)


def analyze(aligned_output, nn_output, nn_representation_of_sentence):
    offset = 0
    complex = []
    simple = []
    for i in range(len(nn_representation_of_sentence)):
        word = nn_representation_of_sentence[i]
        if word == PAR_START or word == SENT_START or word == SENT_END:
            offset += 1
        elif aligned_output[i - offset][0] == '_':
            # word is complex
            complex.append(nn_output[i][0][0])
        else:
            # word is not complex
            simple.append(nn_output[i][0][0])
    return simple, complex


def updateProbs(asent,sent,matches,iarr,arr):
    ''' 
    For each matching word in asent, keep probs ratio, separating
    correct from incorrect.
    '''
    corrprobs = [ ]
    incorrprobs = [ ]
    xcorrprobs = [ ]
    xincorrprobs = [ ]
    if len(asent) != len(matches):
        print "asentXX",asent,"\nmatches",matches
        return corrprobs,incorrprobs        
    probdat = utils.wordStats(arr,iarr,sent) # len of iarr
    #print 'iarr',iarr
    for i in range( len(asent) ):
        idx = matches[i]
        if idx == NO_MATCH: continue
        if asent[i][0] != '_':
            if probdat[idx][0] == 1: # right
                corrprobs.append(probdat[idx][1])
            else:
                incorrprobs.append(probdat[idx][1])
        else:
            if probdat[idx][0] == 1: # right
                xcorrprobs.append(probdat[idx][1])
            else:
                xincorrprobs.append(probdat[idx][1])

    return corrprobs,incorrprobs,xcorrprobs,xincorrprobs


def printPairs(sp):
    for p in sp:
        if p[2]:
            print p[0]
            print p[1]
            print '----'


COUNT = -1
BASEDIR = "/home/af9562/wpred/"
MODEL_NAME = "Best02-srn-63-3.01"
"""
nnetFile = BASEDIR + "data/test/NewselaSimple03test.pbz2"
indexFile = BASEDIR + "data/test/NewselaSimple03test.idx"
"""
"""
nnetFile = BASEDIR + "data/test/NoOverlapRawTest.pbz2"
indexFile = BASEDIR + "data/test/NoOverlapRawTest.idx"
"""
nnetFile = BASEDIR + "data/test/NoOverlapTest.pbz2"
indexFile = BASEDIR + "data/test/NoOverlapTest.idx"
if COUNT != -1:
    probsFile = BASEDIR + MODEL_NAME + "-" + str(COUNT)+"probs.h5"
else:
    probsFile = BASEDIR + MODEL_NAME + "-probs.h5"
get_statistics()
