import numpy
from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
import classpaths as paths
from nltk.corpus import wordnet
from nltk.corpus import cmudict


ORIGINAL_DATA = paths.NEWSELA_ALIGNED + "dataset.txt"
EMBEDDINGS = paths.NEWSELA_ALIGNED + "embeddings_Jul-05-1256_epoch0.tsv"
DENSITIES = paths.NEWSELA_ALIGNED + "density_Jul-05-1256_epoch0.tsv"
FEATURE_DIR = paths.NEWSELA_ALIGNED + "features"
BINARY = lambda x: int(x)

"""
ORIGINAL_DATA = \
    paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt"
EMBEDDINGS = paths.NEWSELA_COMPLEX + "word_embeddings_Jul-05-1256_epoch0.tsv"
DENSITIES = paths.NEWSELA_COMPLEX + "density_Jul-09-1733_epoch0.tsv"
FEATURE_DIR = paths.NEWSELA_COMPLEX + "features"
BINARY = lambda x: 0 if int(x) <= 3  else 1 
"""


FEATURES = {"syllab_sent": count_sent_syllables()}


def cwictorify():
    """
    Converts the original data from the Chris format to CWICTOR format used
    by lexenstein.
    CWICTOR format: sent    word    ind    score (binary)
    :return: the list of lines in CWICTOR format
    """
    lines = get_raw_data()
    return ['\t'.join([x['sent'], x['word'], x['ind'], str(BINARY(x['score']))])
            + '\n' for x in lines]


def get_raw_data():
    """
    Load the raw data (in Chris format) in memory
    :return: list of [word, word_index, complexity, sentence]
    """
    with open(ORIGINAL_DATA) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    lines = [{'word': x[0], 'sent': x[3], 'ind': x[1], 'score':x[2]}
             for x in lines]
    return lines


def count_sent_syllables():
    lines = [x['sent'] for x in get_raw_data()]
    output = numpy.zeros(len(lines))
    for i in range(len(lines)):
        for word in lines[i].split(' '):
            output[i] += syllab_count(word)
    numpy.save(FEATURE_DIR)



def count_sentence_syllables(sent, d = cmudict.dict(), m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)):
    """
    counts the number of syllables in words (strings separated by spaces that
     contain letters) in a  given sentence
    :param sent: the sentence as a string, punctuation separated by spaces
    :return: the number of syllables
    """
    words = sent.split(' ')
    syllables = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            try:
                syllables += count_word_syllables(word, d, m)[0]
            except:
                syllables += count_word_syllables(word, d, m)
        else:
            words.remove(word)
    return float(syllables)/float(len(words))


def count_word_syllables(word, d = cmudict.dict(), m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)):
    """
    Counts the syllables in a word
    :param word: the word to be counted
    :return: the number of syllables
    """
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        return len(m.splitSyllables(word)[0].split('-'))


def calc_sent_len(sent):
    """
    Calculates the number of words in a sentence
    :param sent: the sentence as a string
    :return: the number of words in sent
    """
    words = sent.split(' ')
    length = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            length += 1
    return length


def calc_avg_word_lens(sent):
    """
    Calculates the average length of the words in a sentence
    :param sent: the sentence as a string
    :return: the average number of letters in the words in sent
    """
    words = sent.split(' ')
    totalLen = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            totalLen += len(word)
        else:
            words.remove(word)
    return float(totalLen) / float(len(words))


def calc_syn_avgs(sent):
    """
    Calculates the average number of synonyms of the words in a sentence. Will
    skip words that contain characters unreadable by wordnet
    :param sent: the sentence as a string
    :return: the  average number of synonyms of the words in sent
    """
    words = sent.split(' ')
    totalSyns = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            if not re.match(r'.*[^ -~].*', word):
                for syn in wordnet.synsets(word):
                    totalSyns += len(syn.lemmas())
            else:
                print(word)
        else:
            words.remove(word)
    return float(totalSyns) / float(len(words))


def calc_synset_avgs(sent):
    """
    Calculates the average number of synsets of the words in a sentence. Will
    skip words that contain characters unreadable by wordnet
    :param sent: the sentence as a string
    :return: the  average number of synsets of the words in sent
    """
    words = sent.split(' ')
    totalSets = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            if not re.match('.*[^ -~].*', word):
                totalSets += len(wordnet.synsets(word))
            else:
                print(word)
        else:
            words.remove(word)
    return float(totalSets) / float(len(words))


def calc_nGram_avgs(sent, ngramDict, size):
    """
    calculates the average google 1-gram frequencies of the words in a sentence
    :param sent: the sentence as a string
    :param ngramDict: a dictionary of {word, number of appearances in google
        1-gram}
    :param size: the total number of the appearances of all words in the 1-gram
    :return: he average google 1-gram frequencies of the words in sent
    """
    words = sent.split(' ')
    totalAvg = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            if word in ngramDict:
                totalAvg += float(ngramDict[word]) / size
        else:
            words.remove(word)
    return float(totalAvg) / float(len(words))


def collect_data(corpusPath, CWPath, vecPath, densPath):
    """
    Collects features from a corpus in CWICTOR format from a file at CWPath
    and a file in Kriz format at corpusPath
    :param corpusPath:
    :param CWPath:
    :return: the list of features
    """
    d = cmudict.dict()
    m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)

    fe = FeatureEstimator()
    fe.addLengthFeature('Complexity')  # word length
    fe.addSynonymCountFeature('Simplicity')  # WordNet synonyms
    list = fe.calculateFeatures(cwictorify(), format='cwictor', input='text')

    sentenceSylbs = []
    currentArticle = ""

    with open(CWPath) as out:
        lines = out.readlines()
    with open(corpusPath) as corp:
        orig = corp.readlines()
    with open(densPath) as density:
        densities = density.readlines()
    if USE_WORD_VECS:
        with open(vecPath) as vec:
            vecs = vec.readlines()

    if DEBUG:
        lines = lines[:100]
        orig = orig[:100]
        list = list[:100]
        densities = densities[:100]
        if USE_WORD_VECS:
            vecs = vecs[:100]

    for i in range(len(densities)):
        densities[i] = densities[i].rstrip('\n')
    for i in range(len(vecs)):
        vecs[i] = vecs[i].rstrip('\n')

    sOrig = [j.split('\t') for j in orig]

    # prep 1-gram dictionary
    with open(paths.USERDIR + "/data/web1T/1gms/vocab") as file:
        ngrams = file.readlines()
    for lineNum in range(len(ngrams)):
        ngrams[lineNum] = ngrams[lineNum].split('\t')
    ngramDict = {x[0]: int(x[1]) for x in ngrams}
    size = int(open(paths.USERDIR + "/data/web1T/1gms/total").read())

    # prep graph file
    graphScores = []
    with open(GRAPH_FILE) as file:
        tmp = file.readlines()
        tmp = tmp[1:]
    for lineNum in range(len(tmp)):
        tmp[lineNum] = tmp[lineNum].split('\t')
        graphScores.append(tmp[lineNum][0])
    if DEBUG:
        graphScores = graphScores[:100]

    print("files read")

    # append lines
    for i in range(len(list)):
        # print(i)
        line = lines[i].split('\t')

        # unique WordNet synsets
        if not re.match(r'.*[^ -~].*', line[1]):
            list[i].append(len(wordnet.synsets(line[1])))
        else:
            list[i].append(0)
        # number of syllables
        list[i].append(count_word_syllables(line[1], d, m))
        # google 1-gram freq
        if line[1] in ngramDict:
            # list[i].append(float(ngramDict[line[1]]) / size)
            list[i].append(ngramDict[line[1]])
        else:
            list[i].append(0)

        # graph score
        # list[i].append(graphScores[i])

        # density score
        list[i].append(densities[i].split('\t')[-1])

        # reset sentence features
        index = int(sOrig[i][-1])
        if currentArticle != sOrig[i][-2]:
            currentArticle = sOrig[i][-2]
            sentenceSylbs = []
            sentLens = []
            wordLenAvgs = []
            synonymCountAvgs = []
            synsetNumAvgs = []
            nGramFreqAvgs = []
        # update sentence features
        while len(sentenceSylbs) < index + 1:
            sentenceSylbs.append(count_sentence_syllables(sOrig[i][3], d, m))
            sentLens.append(calc_sent_len(sOrig[i][3]))
            wordLenAvgs.append(calc_avg_word_lens(sOrig[i][3]))
            synonymCountAvgs.append(calc_syn_avgs(sOrig[i][3]))
            synsetNumAvgs.append(calc_synset_avgs(sOrig[i][3]))
            nGramFreqAvgs.append(calc_nGram_avgs(sOrig[i][3], ngramDict, size))

        # number of sentence syllables
        list[i].append(sentenceSylbs[index])
        # sent length
        list[i].append(sentLens[index])
        # avg length of words in sentence
        list[i].append(wordLenAvgs[index])
        # avg synonym count in sentence
        list[i].append(synonymCountAvgs[index])
        # avg num synsets in sentence
        list[i].append(synsetNumAvgs[index])
        # avg word 1-gram freq in sentence
        list[i].append(nGramFreqAvgs[index])

        if USE_WORD_VECS:
            vecvals = vecs[i].split('\t')[1:]
            for val in vecvals:
                list[i].append(val)

        list[i].insert(0, line[2])
        list[i].insert(0, sOrig[i][-1].strip('\n'))
        list[i].insert(0, sOrig[i][-2])
        list[i].append(sOrig[i][0])  # TODO make 10x10 confusion matrix if NNet
        # list.append(line[1])   #causes file to be unreadable?
        if i % 50 == 0:
            print(str(i) + " out of " + str(len(list)))
    return list


def save(data, outPath):
    """
    Saves data to a file at outPath
    :param data:
    :param outPath:
    """
    l =[]
    with open(outPath, 'w') as out:
        out.write('ArticleName SentInd(Article) WordInd(Sentence) WordLength ' +
                  'NumSynonyms NumSynsets WordSyllables 1GramFreq GraphScore AvgSentSylbs SentLength AvgWordLen AvgNumSynonyms AvgNumSynsets Avg1GramFreq\n')
        for line in data:
            s = ''
            for i in range(len(line)-1):
                s += str(line[i]) + '\t'
            s += str(line[len(line)-1])
            l.append(s+'\n')
        out.writelines(l)
    print("Data Saved")