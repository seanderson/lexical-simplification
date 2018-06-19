"""
Classifies words as complex or simple using methods based on Reno Kriz's
Simplification Using Paraphrases and Context-based Lexical Substitution
"""

from lexenstein.identifiers import *
from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
import classpaths as paths


def main():
    m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)

    fe = FeatureEstimator()
    # add features here
    fe.addLengthFeature('Complexity')   # word length
    fe.addSyllableFeature(m, 'Complexity')  # num syllables
    # word freq (from google n-gram)
    # unique WordNet synsets
    # WordNet synonyms

    mli = MachineLearningIdentifier(fe)
    mli.trainSVM()

    labels = mli.identifyComplexWords()


if __name__ == '__main__':
    main()