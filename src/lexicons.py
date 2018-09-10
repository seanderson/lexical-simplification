"""
The module that contains utilities for building and using the lexicons
"""


import sys
import numpy
from lexenstein.util import *

UTAG_MAP = lambda x: getGeneralisedPOS(x)
ANY_POS = ""  # for storing information about any POS
n_unique_words = 0
n_unique_entries = 0


def build_ultimate_lexicon(ul, lexicon, n_of_lexicons, l_id, l_name):
    """
    Add words in lexicon to the ul (ultimate lexicon)
    :param ul:            the ultimate lexicon which is being built
    :param lexicon:       the lines from which to build the ul
    :param n_of_lexicons: number of different lexicons used to build the ul
    :param l_id:          the index of this particular lexicon among these
    :param l_name:        the name of te hlexicon
    :return: None
    """
    global n_unique_words
    global n_unique_entries
    for i in range(len(lexicon)):
        if len(lexicon[i].rstrip('\n').split('\t')) != 2:
            print("Lexicon: " + l_name + ", line: " + str(i))
            print("Too many values to unpack: " + lexicon[i])
            exit(-1)
        word, tag = lexicon[i].rstrip('\n').split('\t')
        word = word.lower()
        """if tag not in UTAG_MAP:
            print("Lexicon: " + l_name + ", line: " + str(i))
            print("Word " + word + " is marked with an unknown POS tag: " + tag)
            exit(-1)"""
        if word not in ul:
            n_unique_words += 1
            ul[word] = {}
            ul[word][ANY_POS] = numpy.zeros(n_of_lexicons)
        tag = UTAG_MAP(tag)
        if tag not in ul[word]:
            n_unique_entries += 1
            ul[word][tag] = numpy.zeros(n_of_lexicons)
        ul[word][tag][l_id] = 1
        ul[word][ANY_POS][l_id] = 1


def write_ultimate_lexicon(ul, filename, lex_names):
    """
    Write the ultimate lexicon to a file (.tsv format)
    :param ul:        the ultimate lexicon
    :param filename:  the name of the file to which to write the ul
    :param lex_names: the header
    :return: None
    """
    with open(filename, 'w') as file:
        file.write('Word' + '\t' + '\t'.join(lex_names))
        for word in sorted(ul.keys()):
            for tag in ul[word].keys():
                if tag != "":
                    file.write('\n' + word + '_' + tag + '\t' +
                               '\t'.join([str(x) for x in ul[word][tag].tolist()]))
                else:
                    file.write('\n' + word + '\t' + '\t'.join([str(x) for x in ul[word][tag].tolist()]))


def load_ultimate_lexicons(filename):
    """
    Load the ultimate lexicon into memory
    :param filename: name of teh file from which to load teh lexicon
    :return: the ultimate lexicon
    """
    ul = {}
    with open(filename) as file:
        for line in file:
            line = line.rstrip('\n').split('\t')
            entry = line[0]
            features = numpy.array(int(x) for x in line[1:])
            if entry in ul:
                print("The word " + entry + " is already in the lexicon")
            ul[entry] = features
    return ul


def get_word_features(word, ul):
    """
    Get word features from the ul
    :param word:
    :param ul:
    :return: A numpy array of size get_n_of_features containing 1s and 0s
    """
    word = word.lower().split('_')
    tag = word[-1]
    word = '_'.join(word[:-1]) + '_'
    if word in ul:
        if word + '_' + tag in ul:
            return numpy.concatenate((ul[word], ul[word + tag]))
        else:
            return numpy.concatenate((ul[word], numpy.zeros(len(ul[word]))))
    else:
        return numpy.zeros(get_n_of_features(ul))


def get_n_of_features(ul):
    """
    Return the length hof the numpy array returned by get_word_features
    :param ul:
    :return:
    """
    return len(ul[ul.keys()[0]] * 2)


if __name__ == "__main__":
    ul = {}  # ultimate lexicon
    if len(sys.argv) != 3:
        print("please give (only) two arguments: the name of the metafile with "
              "all the lexicons listed and the name of the file to write the "
              "ultimate lexicon to")
        exit(-1)
    with open(sys.argv[1]) as metafile:
        files = metafile.readlines()
    lex_names = []
    for i in range(len(files)):
        filename, l_name = files[i].rstrip('\n').split('\t')
        lex_names.append(l_name)
        with open(filename) as file:
            build_ultimate_lexicon(ul, file.readlines(), len(files), i, l_name)
    print("Total number of entries (excluding words without tags): "
          + str(n_unique_entries))
    print("Total number of unique words: " + str(n_unique_words))
    print("Average number of POS tags per word: " + str(
        round(float(n_unique_entries) / n_unique_words, 2)))
    write_ultimate_lexicon(ul, sys.argv[2], lex_names)
