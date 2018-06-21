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


def save(data, outPath):
    l =[]
    with open(outPath, 'w') as out:
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


def collect_data(corpus, output):
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

    with open(output) as out:
        lines = out.readlines()
    with open(corpus) as corp:
        orig = corp.readlines()

    # prep 1-gram dictionary
    with open(paths.USERDIR + "/data/web1T/1gms/vocab") as file:
        ngrams = file.readlines()
    for lineNum in range(len(ngrams)):
        ngrams[lineNum] = ngrams[lineNum].split('\t')
    ngramDict = {x[0]: int(x[1]) for x in ngrams}
    size = int(open(paths.USERDIR + "/data/web1T/1gms/total").read())

    for i in range(len(list)):
        line = lines[i].split('\t')
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
        list[i].insert(0, sOrig[i][0])

    '''data = []
    data.append([line.split('\t')[0].split(' ') for line in lines])
    count_sentence_syllables(data, m)'''

    return list


def main(corpus, output):
    return collect_data(corpus, output)


if __name__ == '__main__':
    # main('train_cwictor_corpus.txt', 'test_cwictor_corpus.txt')
    data = (main(paths.NEWSELA_COMPLEX +
               "Newsela_Complex_Words_Dataset_supplied.txt", paths.NEWSELA_COMPLEX+"Cwictorified"))
    save(data, paths.NEWSELA_COMPLEX + "testFeatClass.txt")
    print (data)