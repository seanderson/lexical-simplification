"""
Classifies words as complex or simple using methods based on Reno Kriz's
Simplification Using Paraphrases and Context-based Lexical Substitution
"""

from lexenstein.identifiers import *
from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
import classpaths as paths
from nltk.corpus import wordnet


def cwictorify(corpus):
    # format: Sentence   word    indexInSent     BinaryIsComplex
    input = open(corpus)
    output = open()
    for line in input:
        list = line.split('\t')
        print(line)

    return output



def main(corpus):
    m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)

    fe = FeatureEstimator()
    # add features here
    fe.addLengthFeature('Complexity')   # word length
    fe.addSyllableFeature(m, 'Complexity')  # num syllables
    # word freq (from google n-gram)
    # unique WordNet synsets
    # WordNet synonyms
    return fe.calculateFeatures(cwictorify(corpus), format='cwictor')


if __name__ == '__main__':
    # main('train_cwictor_corpus.txt', 'test_cwictor_corpus.txt')
    print(main(paths.NEWSELA_COMPLEX +
               "Newsela_Complex_Words_Dataset_supplied.txt"))
