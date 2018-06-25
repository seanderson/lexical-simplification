"""
Classifies words as complex or simple using methods based on Reno Kriz's
Simplification Using Paraphrases and Context-based Lexical Substitution
"""

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


def cwictorify(inputPath, outputPath):
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
    l =[]
    with open(outPath, 'w') as out:
        out.write('SentInd(Article)\tWordInd(Sentence)\tWordLength\tWordSyllables\tNumSynonyms\tNumSynsets\t1GramFreq\n')
        for line in data:
            s = ''
            for i in range(len(line)-1):
                s += str(line[i]) + '\t'
            s += str(line[len(line)-1])
            l.append(s+'\n')
        out.writelines(l)


def count_sentence_syllables(data, mat):
    input = []
    for line in data:
        for subst in line[3:len(line)]:
            word = subst.strip().split(':')[1].strip()
            input.append(word)


def count_word_syllables(word):
    d = cmudict.dict()
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
    except:
        m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)
        return len(m.splitSyllables(word)[0].split('-'))


def collect_data(corpus, output):
    """

    :param corpus:
    :param output:
    :return: -1 in list[3] if the word contains a non-ASCII char
    """
    m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)

    fe = FeatureEstimator()
    fe.addLengthFeature('Complexity')  # word length
    #fe.addSyllableFeature(m, 'Complexity')  # num syllables
    fe.addSynonymCountFeature('Simplicity')  # WordNet synonyms
    list = fe.calculateFeatures(cwictorify(corpus, output), format='cwictor')

    with open(output) as out:
        lines = out.readlines()
    with open(corpus) as corp:
        orig = corp.readlines()

    if(DEBUG):
        lines = lines[:100]
        orig = orig[:100]
        list = list[:100]

    # prep 1-gram dictionary
    with open(paths.USERDIR + "/data/web1T/1gms/vocab") as file:
        ngrams = file.readlines()
    for lineNum in range(len(ngrams)):
        ngrams[lineNum] = ngrams[lineNum].split('\t')
    ngramDict = {x[0]: int(x[1]) for x in ngrams}
    size = int(open(paths.USERDIR + "/data/web1T/1gms/total").read())

    for i in range(len(list)):
        #print(i)
        line = lines[i].split('\t')
        # number of syllables
        try:
            list[i].append(count_word_syllables(line[1])[0])
        except:
            list[i].append(count_word_syllables(line[1]))
        # unique WordNet synsets
        if not re.match(r'.*[^ -~].*', line[1]):
            list[i].append(len(wordnet.synsets(line[1])))
        else:
            list[i].append(-1)
        # google 1-gram freq
        if line[1] in ngramDict:
            list[i].append(float(ngramDict[line[1]]) / size)
            # list[i].append(ngramDict[line[1]])
        else:
            list[i].append(-1)
        sOrig = [j.split('\t') for j in orig]
        list[i].insert(0, line[2])
        list[i].insert(0, sOrig[i][-1].strip('\n'))
        list[i].insert(0, sOrig[i][-2])
        list[i].append(sOrig[i][0])
        # list.append(line[1])   #causes file to be unreadable?

    '''data = []
    data.append([line.split('\t')[0].split(' ') for line in lines])
    count_sentence_syllables(data, m)'''

    return list


def read_features(filepath):
    data = []
    with open(filepath) as file:
        lines = file.readlines()
    for line in lines:
        data.append(line.split('\t')[2:-1])
    return data[1:]


def read_complexities(filepath):
    complexities = []
    with open(filepath) as file:
        lines = file.readlines()
    for line in lines:
        complexities.append(line.split('\t')[2])
    return complexities


def classify(data):
    labels = numpy.zeros(len(data))
    for i in range(len(labels)):
        labels[i] = i
    clf = svm.SVC(kernel='linear')
    clf.fit(data[0], data[1])
    return clf


def test_classify(X, Y):
    check = []
    if len(X) != len(Y):
        return -1
    if DEBUG:
        X = X[:200]
        Y = Y[:200]
    numTrain = int(.80 * len(X))
    numTimesToTest = 100
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
        if i == True:
            numRight += 1
    return float(numRight)/float(len(check))


def main(corpus, output):
    return collect_data(corpus, output)


if __name__ == '__main__':
    if(TESTCLASSIFY):
        iris = datasets.load_iris()
        print(test_classify(iris.data, iris.target))
    if(CWICTORIFY):
        cwictorify(paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt",
               paths.NEWSELA_COMPLEX+"Cwictorified")
    if(IMPORTDATA):
        data = (main(paths.NEWSELA_COMPLEX +
               "Newsela_Complex_Words_Dataset_supplied.txt", paths.NEWSELA_COMPLEX+"Cwictorified"))
        save(data, paths.NEWSELA_COMPLEX + "testFeatClass.txt")
        print (data)
    featureData = read_features(paths.NEWSELA_COMPLEX + "testFeatClass.txt")
    complexScores = read_complexities(paths.NEWSELA_COMPLEX +
                                        "Newsela_Complex_Words_Dataset_supplied.txt")
    #clf = classify([featureData, complexScores])
    print(test_classify(featureData, complexScores))
