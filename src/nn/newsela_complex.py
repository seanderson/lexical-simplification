"""
Code for tagging the Chris paper data and computing word embeddings for
different words in it.
"""

import sys
from gensim.models import word2vec
from nn.globdefs import *
from nn.word2vecpos import EpochSaver
sys.path.append("../")
import StanfordParse
import re


CHRIS_DATA = "/home/nlp/corpora/newsela_complex/" \
             "Newsela_Complex_Words_Dataset_supplied.txt"
MODEL_FILE = "/home/nlp/newsela/src/nn/cbow-2018-Jul-05-1256/epoch0.model"
OUTPUT_FILE = "/home/nlp/corpora/newsela_complex/" \
              "density_Jul-05-1256_epoch0.tsv"
EMBED_SIZE = 1300
PREFIX = "SDGHKASJDGHKJA"  # A string that should not appear in the test itself
DENSITY_LEVELS = [1, 5, 10, 50, 100, 200]
COSINE_LEVELS = [0.4, 0.3, 0.25]
# The topN numbers to consider when calculating the density measure


def tag_chris_data(filename):
    """
    A function that tags the Chris data and adds special PREFIX to every new
    line so that the original order of lines can be preserved
    :return:
    """
    with open(filename) as file:
        lines = file.readlines()
    with open(filename + ".sents", "w") as file:
        for line in lines:
            file.write(PREFIX + " " + line.split('\t')[3].rstrip('\n') + '\n')
    StanfordParse.tag(filename + ".sents")
    with open(filename + ".sents.tagged") as file:
        lines_tagged = file.readlines()

    # Restoring the original line order in the .tagged file
    j = 0
    while j < len(lines_tagged):
        if lines_tagged[j][:len(PREFIX)] == PREFIX:
            lines_tagged[j] = ' '.join(lines_tagged[j].split(' ')[1:])
            j += 1
        else:
            lines_tagged[j - 1] = lines_tagged[j - 1].rstrip(' \n') + ' ' + \
                                  lines_tagged[j]
            del lines_tagged[j]

    if len(lines) != len(lines_tagged):
        print("File lengths are unequal!")
        exit(-1)

    with open(filename + ".sents.tagged", 'w') as file:
        file.write('\n'.join([x.rstrip('\n') for x in lines_tagged]) + '\n')


def get_tagged_data(lines, lines_tagged):
    """
    Get the lines from teh Chris paper and teh lines from the tagged versino of
    the same data and return a list of words (one per line) in a form that
    should be accepted by the model
    :param lines:
    :param lines_tagged:
    :return:
    """
    final_lines = []
    for i in range(len(lines)):
        word = lines[i].split('\t')[0].casefold()
        ind = int(lines[i].split('\t')[1].casefold())
        line = lines[i].split('\t')[3].rstrip('\n').casefold()
        if re.match('.*f331e.s3.amazonaws.com.*?&gt ; .*', line) and ind > 0:
            ind -= 23
        line = re.sub('.*f331e.s3.amazonaws.com.*?&gt ; ', '', line).split(' ')
        line_tagged = lines_tagged[i].rstrip('\n').casefold()
        line_tagged = re.sub('.*f331e.s3.amazonaws.com.*?&_cc gt_nn ;_: ', '',
                             line_tagged)
        line_tagged = re.sub('a\.m\._nn \._\.', 'a.m._nn',
                             line_tagged)
        line_tagged = re.sub('u\.s_nnp \._\.', 'u.s._nnp', line_tagged).split(
            ' ')
        if '\xa0' in line:
            line_tagged.insert(line.index('\xa0'), '\xa0')
        if len(line) != len(line_tagged):
            print(line)
            print(line_tagged)
            print("Line lengths are unequal! ln:" + str(i))
            exit(-1)
        if word != line[ind]:
            if word == line[ind - 2]:
                ind -= 2
            else:
                print(word + " " + str(ind))
                print(line[ind])
                print("Inconsistency withing the line!" + str(i))
                exit(-1)
        if word != '_'.join(line_tagged[ind].split('_')[:-1]):
            if re.sub('&amp;', '&', word) != '_'.join(
                    line_tagged[ind].split('_')[:-1]):
                word = re.sub('&amp;', '&', word)
                print(word + " " + str(ind))
                print(line_tagged[ind])
                print("Inconsistency withing the tagged line!")
                exit(-1)
        tag = line_tagged[ind].split('_')[-1]
        tag = UTAG_MAP.get(tag, tag)
        final_lines.append(word + '_' + tag)
    return final_lines


def is_sorted(list):
    """
    Returns True if the list is sorted in non ascending order
    :param list:
    :return:
    """
    for i in range(len(list) - 1):
        if list[i] < list[i+1]:
            return False
    return True


def process_chris_data(filename, model, output_name, emb_size, EMBEDDINGS=True,
                       DENSITY=False, COSINE=True):
    """
    Open a pretagged chris file and output word_embeddings and/or density
    measurements
    :param filename:
    :param model:
    :param output_name:
    :param emb_size:
    :param EMBEDDINGS: if True, store embeddings in a file
    :param DENSITY: if True, store density measurements in a file
    :return:
    """
    if not EMBEDDINGS and not DENSITY:
        print("Either EMBEDDINGS or DENSITY must be True")
        exit(-1)

    with open(filename) as file:
        lines = file.readlines()
    with open(filename + ".sents.tagged") as file:
        lines_tagged = file.readlines()

    final_lines = get_tagged_data(lines, lines_tagged)
    print("Check_completed")
    model = word2vec.Word2Vec.load(model)
    print("Model_loaded")

    with open(output_name, 'w') as out_file:
        for word in final_lines[:10]:
            if word not in model.wv.vocab:
                print("WW: word not in vocabulary: " + word)
                vector = "0\t" * (emb_size - 1) + "0"
                if COSINE:
                    density = "0\t" * (len(COSINE_LEVELS) - 1) + "0"
                else:
                    density = "0\t" * (len(DENSITY_LEVELS) - 1) + "0"
            else:
                vector = '\t'.join([str(x) for x in model.wv.get_vector(word)])
                similar = model.wv.most_similar(positive=[word], topn=DENSITY_LEVELS[-1])
                similar = [x[1] for x in similar]
                if not is_sorted(similar):
                    print("List must be sorted")
                    exit(-1)
                if COSINE:
                    density = []
                    for i in range(len(COSINE_LEVELS)):
                        density.append(sum(x >= COSINE_LEVELS[i] for x in similar))
                        j = 2
                        while density[-1] == len(similar):
                            similar = model.wv.most_similar(positive=[word],
                                                            topn=DENSITY_LEVELS[
                                                                 -1] * j)
                            similar = [x[1] for x in similar]
                            j += 1
                            density[-1] = sum(x >= COSINE_LEVELS[i] for x in similar)

                else:
                    density = [str(float(sum(similar[:d]))/d) for d in DENSITY_LEVELS]
                    density = '\t'.join(density)
            out_line = word
            if EMBEDDINGS:
                out_line += '\t' + vector
            if DENSITY:
                out_line += '\t' + density
            out_file.write(out_line + '\n')


if __name__ == "__main__":
    # tag_chris_data(CHRIS_DATA)
    process_chris_data(CHRIS_DATA, MODEL_FILE, OUTPUT_FILE, EMBED_SIZE,
                       EMBEDDINGS=False, DENSITY=True)