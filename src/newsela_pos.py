"""
Code for POS tagging of newsela data in Chris format.
Note: the function that should be used is "get_tags"
"""

import StanfordParse
import re
from lexenstein.util import getGeneralisedPOS

PREFIX = "SasHKADGKJA"  # A string that should not appear in the text itself
TAGGING_DIRECTORY = "/home/nlp/corpora/newsela_aligned/"
# a directory there all the tagged files go


def tag_data(lines):
    """
    A function that tags the Chris data and adds special PREFIX to every new
    line so that the original order of lines can be preserved
    :return:
    """
    with open(TAGGING_DIRECTORY + "tmp.txt", "w") as file:
        for line in lines:
            file.write(PREFIX + " " + line['sent'] + '\n')
    StanfordParse.tag(TAGGING_DIRECTORY + "tmp.txt")
    with open(TAGGING_DIRECTORY + "tmp.txt.tagged") as file:
        lines_tagged = file.readlines()
    # Restoring the original line order in the .tagged file
    j = 0
    while j < len(lines_tagged):
        if lines_tagged[j][:len(PREFIX)] == PREFIX:
            lines_tagged[j] = ' '.join(lines_tagged[j].split(' ')[1:])
            if PREFIX in [x.split('_')[0] for x in lines_tagged[j].split(' ')]:
                line = lines_tagged[j].split(' ')
                ind = [x.split('_')[0] for x in
                       lines_tagged[j].split(' ')].index(PREFIX)
                lines_tagged[j] = ' '.join(line[:ind]).rstrip('\n') + '\n'
                newline = ' '.join(line[ind:])
                lines_tagged.insert(j+1, newline)
            j += 1
        else:
            lines_tagged[j - 1] = PREFIX + " " + lines_tagged[
                j - 1].rstrip(' \n') + ' ' + lines_tagged[j]
            del lines_tagged[j]
            j -= 1

    if len(lines) != len(lines_tagged):
        print("File lengths are unequal!")
        exit(-1)

    return [x.rstrip('\n') for x in lines_tagged]


def verify_data(lines, lines_tagged):
    """
    Get the lines from the Chris paper and teh lines from the tagged versino of
    the same data and return a list of words (one per line) in a form that
    should be accepted by the model
    :param lines:
    :param lines_tagged:
    :return:
    """
    final_lines = []
    for i in range(len(lines)):
        if not lines[i]['words']:
            final_lines.append('PHRASE')
            continue
        word = lines[i]['words'][0].lower()
        ind = lines[i]['inds'][0]
        line = lines[i]['sent'].lower()
        if re.match('.*f331e.s3.amazonaws.com.*?&gt ; .*', line) and ind > 0:
            ind -= 23
        line = re.sub('.*f331e.s3.amazonaws.com.*?&gt ; ', '', line).split(' ')
        line_tagged = lines_tagged[i].rstrip('\n').lower()
        line_tagged = re.sub('.*f331e.s3.amazonaws.com.*?&_cc gt_nn ;_: ', '',
                             line_tagged)
        line_tagged = re.sub('a\.m\._nn \._\.', 'a.m._nn',
                             line_tagged)
        line_tagged = re.sub('d-ill_nnp \._\.', 'd_ill._nnp',
                             line_tagged)
        line_tagged = re.sub('u\.s_nnp \._\.', 'u.s._nnp', line_tagged).split(
            ' ')
        if '\xa0' in line:
            line_tagged.insert(line.index('\xa0'), '\xa0')
        if len(line) != len(line_tagged):
            line_tagged = re.sub(r' `_`` (s|re|ll|d|ve|m|60s|t|80s|40s)_',
                                 r' `\1_', ' '.join(line_tagged)).split(' ')
            if len(line) != len(line_tagged):
                print("Line lengths are unequal! ln:" + str(i))
                exit(-1)
        if word != line[ind]:
            if word == line[ind - 2]:
                ind -= 2
            else:
                print("Inconsistency withing the line!" + str(i))
                exit(-1)
        if word != '_'.join(line_tagged[ind].split('_')[:-1]):
            if re.sub('&amp;', '&', word) != '_'.join(
                    line_tagged[ind].split('_')[:-1]):
                print("Inconsistency withing the tagged line!")
                exit(-1)
        tag = line_tagged[ind].split('_')[-1].upper()
        tag = getGeneralisedPOS(tag)
        final_lines.append(tag)
    return final_lines


def get_tags(text):
    """
    A function that gets a list of lines (Chris paper format)
    and returns a list of tags (for target lines)
    :param text:
    :return:
    """
    return verify_data(text, tag_data(text))
