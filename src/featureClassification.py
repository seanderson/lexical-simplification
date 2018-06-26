"""
Classifies words as complex or simple using methods based on Reno Kriz's
Simplification Using Paraphrases and Context-based Lexical Substitution
"""
# TODO squish data from 0-1
# TODO make features more modular
import re
from lexenstein.identifiers import *
from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
import classpaths as paths
from nltk.corpus import wordnet
from nltk.corpus import cmudict
from sklearn import datasets
from sklearn import svm
import copy
import random


CWICTORIFY = False
TESTCLASSIFY = False
IMPORTDATA = False
DEBUG = False

NEWSELLA_SUPPLIED = paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt"
CWICTOIFIED = paths.NEWSELA_COMPLEX + "Cwictorified.txt"
SAVE_FILE = paths.NEWSELA_COMPLEX + "Feature_data.txt"
GRAPH_FILE = paths.NEWSELA_COMPLEX + "Graph_output.txt"


def cwictorify(inputPath, outputPath):
    """
    Writes the file from inputPath in CWICTOR format
    :param inputPath:
    :param outputPath:
    """
    # format: Sentence   word    indexInSent     BinaryIsComplex
    with open(inputPath) as file:
        input = file.readlines()
    with open(outputPath,"w") as output:
        for line in input:
            list = line.split('\t')
            #print(list)
            if list[2] > 3:
                c = 1
            else:
                c = 0
            output.write(list[3]+"\t"+list[0]+"\t"+list[1]+"\t"+str(c)+"\n")
    return outputPath


def save(data, outPath):
    """
    Saves data to a file at outPath
    :param data:
    :param outPath:
    """
    l =[]
    with open(outPath, 'w') as out:
        out.write('SentInd(Article)\tWordInd(Sentence)\tWordLength\tNumSynonyms\tNumSynsets\tWordSyllables\t1GramFreq\n')
        for line in data:
            s = ''
            for i in range(len(line)-1):
                s += str(line[i]) + '\t'
            s += str(line[len(line)-1])
            l.append(s+'\n')
        out.writelines(l)
    print("Data Saved")


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


def collect_data(corpusPath, CWPath):
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
    list = fe.calculateFeatures(cwictorify(corpusPath, CWPath), format='cwictor')

    sentenceSylbs = []
    currentArticle = ""

    with open(CWPath) as out:
        lines = out.readlines()
    with open(corpusPath) as corp:
        orig = corp.readlines()

    if(DEBUG):
        lines = lines[:100]
        orig = orig[:100]
        list = list[:100]

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
        tmp = tmp [1:]
    for lineNum in range(len(tmp)):
        tmp[lineNum] = tmp[lineNum].split('\t')
        graphScores.append(tmp[lineNum][0])

    print("files read")

    for i in range(len(list)):
        #print(i)
        line = lines[i].split('\t')
        # unique WordNet synsets
        if not re.match(r'.*[^ -~].*', line[1]):
            list[i].append(len(wordnet.synsets(line[1])))
        else:
            list[i].append(-1)
        #print("syn done")
        # number of syllables
        list[i].append(count_word_syllables(line[1], d, m))
        #print("sylbs done")
        # number of sentence syllable
        index = int(sOrig[i][-1])
        if currentArticle != sOrig[i][-2]:
            currentArticle = sOrig[i][-2]
            sentenceSylbs = []
        while len(sentenceSylbs) < index+1:
            sentenceSylbs.append(count_sentence_syllables(sOrig[i][3], d, m))
        list[i].append(sentenceSylbs[index])
        #print("sent sylbs done")
        # google 1-gram freq
        if line[1] in ngramDict:
            list[i].append(float(ngramDict[line[1]]) / size)
            # list[i].append(ngramDict[line[1]])
        else:
            list[i].append(-1)
        #print("ngram done")
        # graph score
        list[i].append(graphScores[i])

        list[i].insert(0, line[2])
        list[i].insert(0, sOrig[i][-1].strip('\n'))
        list[i].insert(0, sOrig[i][-2])
        list[i].append(sOrig[i][0])
        # list.append(line[1])   #causes file to be unreadable?
        if i % 50 == 0:
            print(str(i) + " out of " + str(len(list)))

    '''data = []
    data.append([line.split('\t')[0].split(' ') for line in lines])
    count_sentence_syllables(data, m)'''

    return list


def read_features(filepath):
    """
    Reads features from a file created with the save() function
    :param filepath: the path to the file to read features from
    :return: a list of the features
    """
    data = []
    with open(filepath) as file:
        lines = file.readlines()
    for line in lines:
        data.append(line.split('\t')[2:-1])
    return data[1:]


def read_complexities(filepath):
    """
    reads Kriz complexity scores from a file at filepath
    :param filepath:
    :return: a list of complexity scores from the file at filepath
    """
    complexities = []
    with open(filepath) as file:
        lines = file.readlines()
    for line in lines:
        complexities.append(line.split('\t')[2])
    return complexities


def classify(data):
    """
    trains a SVM on data
    :param data: the data to trian the SVM on. In format [X,Y]
    :return: the trained SVM
    """
    labels = numpy.zeros(len(data))
    for i in range(len(labels)):
        labels[i] = i
    clf = svm.SVC(cache_size= 500, kernel='rbf')
    clf.fit(data[0], data[1])
    return clf


def test_classify(X, Y):
    """
    Tests the SVM
    :param X: feature data
    :param Y: label data
    :return: a decmal frequency of correctly predicted answers
    """
    check = []
    if len(X) != len(Y):
        return -1
    if DEBUG:
        X = X[:200]
        Y = Y[:200]
    numTrain = int(.80 * len(X))
    numTimesToTest = 1
    for i in range(numTimesToTest):
        available = [copy.copy(X), copy.copy(Y)]
        train = [[], []]
        for j in range(numTrain):
            index = random.randint(0, len(available[0])-1)
            train[0].append(available[0][index])
            train[1].append(available[1][index])
            numpy.delete(available[0], index)
            numpy.delete(available[1], index)
        test = available
        clf = classify(train)
        preds = clf.predict(test[0])
        print("Testing: "+str(i)+" Out of "+str(numTimesToTest))
        for j in range(len(test[0])):
            check.append(preds[j] == test[1][j])
    numRight = 0
    for i in check:
        if i:
            numRight += 1
    return float(numRight)/float(len(check))


def main(corpus, output):
    return collect_data(corpus, output)


if __name__ == '__main__':
    if(TESTCLASSIFY):
        iris = datasets.load_iris()
        print(test_classify(iris.data, iris.target))
    if(CWICTORIFY):
        cwictorify(NEWSELLA_SUPPLIED, CWICTOIFIED)
    if(IMPORTDATA):
        data = (main(NEWSELLA_SUPPLIED, CWICTOIFIED))
        save(data, SAVE_FILE)
        #print (data)
    featureData = read_features(SAVE_FILE)
    complexScores = read_complexities(NEWSELLA_SUPPLIED)
    #clf = classify([featureData, complexScores])
    print(test_classify(featureData, complexScores))
