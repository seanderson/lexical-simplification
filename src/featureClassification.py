"""
Classifies words as complex or simple using methods based on Reno Kriz's
Simplification Using Paraphrases and Context-based Lexical Substitution
"""

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


def main(corpus, output):
    m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)

    fe = FeatureEstimator()
    # add features here
    fe.addLengthFeature('Complexity')   # word length
    fe.addSyllableFeature(m, 'Complexity')  # num syllables
    # word freq (from google n-gram)
    # unique WordNet synsets
    # WordNet synonyms
    return fe.calculateFeatures(cwictorify(corpus,output), format='cwictor')


if __name__ == '__main__':
    # main('train_cwictor_corpus.txt', 'test_cwictor_corpus.txt')
    print(main(paths.NEWSELA_COMPLEX +
               "Newsela_Complex_Words_Dataset_supplied.txt", paths.NEWSELA_COMPLEX+"Cwictorified"))
