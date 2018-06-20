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


def collectData(corpus, output):
    """

    :param corpus:
    :param output:
    :return: -1 in list[3] if the word contains a non-ASCII char
    """
    m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)


    fe = FeatureEstimator()
    fe.addLengthFeature('Complexity')  # word length
    fe.addSyllableFeature(m, 'Complexity')  # num syllables
    fe.addSynonymCountFeature('Simplicity')  # WordNet synonyms
    list = fe.calculateFeatures(cwictorify(corpus, output), format='cwictor')

    # unique WordNet synsets
    with open(output) as out:
        lines = out.readlines()
    for i in range(len(list)):
        line = lines[i].split('\t')
        if not re.match(r'.*[^ -~].*', line[1]):
            list[i].append(len(wordnet.synsets(line[1])))
        else:
            list[i].append(-1)

    # google 1-gram frequency
    with open(paths.USERDIR + "/data/web1T/1gms/vocab") as file:
        ngrams = file.readlines()
    for lineNum in range(len(ngrams)):
        ngrams[lineNum] = ngrams[lineNum].split('\t')
    ngramDict = {x[0]: int(x[1]) for x in ngrams}
    size = int(open(paths.USERDIR + "/data/web1T/1gms/total").read())
    for i in range(len(list)):
        line = lines[i].split('\t')
        if(line[1] in ngramDict):
            list[i].append(float(ngramDict[line[1]])/size)
            #list[i].append(ngramDict[line[1]])
        else:
            list[i].append(-1)

    return list


def main(corpus, output):
    return collectData(corpus, output)


if __name__ == '__main__':
    # main('train_cwictor_corpus.txt', 'test_cwictor_corpus.txt')
    print(main(paths.NEWSELA_COMPLEX +
               "Newsela_Complex_Words_Dataset_supplied.txt", paths.NEWSELA_COMPLEX+"Cwictorified"))
