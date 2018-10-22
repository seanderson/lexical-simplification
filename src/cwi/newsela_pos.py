"""
Code for POS tagging of newsela data in Chris format.
Note: the function that should be used is "get_tags"
"""

import re
from src.word2vec.globdefs import getGeneralisedPOS
from src import classpaths as path
import subprocess

PREFIX = "SasHKADGKJA"  # A string that should not appear in the text itself
TAGGING_DIRECTORY = "/home/nlp/corpora/newsela_aligned/"
# a directory there all the tagged files go

def tag(textfile):
    result = subprocess.check_output(['java', '-mx2048m','-cp', path.CLASSPATH, "edu.stanford.nlp.tagger.maxent.MaxentTagger", "-model", '/home/nlp/wpred/stanford-postagger/models/english-bidirectional-distsim.tagger', "-textFile", textfile], shell=False)
    with open(textfile + ".tagged", 'w') as file:
        file.writelines(result.decode("utf8"))

def tag_data(lines):
    """
    A function that tags the Kriz data and adds special PREFIX to every new
    line so that the original order of lines can be preserved
    :return:
    """
    with open(TAGGING_DIRECTORY + "tmp.txt", "w") as file:
        for line in lines:
            file.write(PREFIX + " " + line['sent'] + '\n')
    tag(TAGGING_DIRECTORY + "tmp.txt")
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
    Get the lines from the Chris paper and the lines from the tagged versino of
    the same data and return a list of words (one per line) in a form that
    should be accepted by the model
    :param lines:
    :param lines_tagged:
    :return:
    """
    final_lines = []
    for i in range(len(lines)):
        final_lines.append([])
        words = lines[i]['sent'].lower().split(' ')
        words = [(words[j], j) for j in range(len(words))]
        if not lines[i]['phrase']:
            words.append((lines[i]['words'][0].lower(), lines[i]['inds'][0]))
        for word, ind in words:
            line = lines[i]['sent'].lower()
            if re.match('.*f331e.s3.amazonaws.com.*?&gt ; .*', line) and ind > 0:
                ind -= 20
            line = re.sub('.*f331e.s3.amazonaws.com.*?&gt ; ', '', line).split(' ')
            line_tagged = lines_tagged[i].rstrip('\n').lower()
            line_tagged = re.sub('.*f331e.s3.amazonaws.com.*?&_cc gt_nn ;_: ',
                                 '', line_tagged)
            line_tagged = re.sub('a\.m\._nn \._\.', 'a.m._nn', line_tagged)
            line_tagged = re.sub('d-ill_nnp \._\.', 'd_ill._nnp', line_tagged)
            line_tagged = re.sub('\xc2\xa0', ' ', line_tagged)
            line_tagged = re.sub('u\.s_nnp \._\.', 'u.s._nnp',
                                 line_tagged).split(' ')
            if '\xa0' in line:
                line_tagged.insert(line.index('\xa0'), '\xa0')
            if len(line) != len(line_tagged):
                line_tagged = re.sub(r' `_`` (s|re|ll|d|ve|m|60s|t|80s|40s)_',
                                     r' `\1_', ' '.join(line_tagged)).split(' ')
                if len(line) != len(line_tagged):
                    # print(line)
                    # print(line_tagged)
                    # print("Line lengths are unequal! ln:" + str(i))
                    final_lines[i].append(get_tag_brute_force(line_tagged, word))
                    continue
            if word != line[ind]:
                if word == line[ind - 2]:
                    ind -= 2
                else:
                    # print(word)
                    # print(line[ind])
                    # print(line)
                    # print("Inconsistency withing the line!" + str(i))
                    final_lines[i].append(get_tag_brute_force(line_tagged, word))
                    continue
            if word != '_'.join(line_tagged[ind].split('_')[:-1]):
                if re.sub('&amp;', '&', word) != '_'.join(
                        line_tagged[ind].split('_')[:-1]):
                    # print("Inconsistency withing the tagged line!")
                    final_lines[i].append(get_tag_brute_force(line_tagged, word))
                    continue
            tag = line_tagged[ind].split('_')[-1].upper()
            tag = getGeneralisedPOS(tag)
            final_lines[i].append(tag)
    return final_lines


def tokenize(sent, ind):
    """
    Imitation of stanford tokenizer that gets a character index and makes
    sure that it point to the same char after tokenizatino
    :param sent:
    :param ind:
    :return: sent, ind
    """
    sent = re.sub(r'\xe2\x80.', ' ', sent)
    j = 0
    while j < len(sent):
        if sent[j] in ',":(][)$?;!.':
            if j < ind:
                ind += 1
            if j != len(sent) - 1:
                if sent[j - 1] not in ' "' and sent[j + 1] not in ' ",' or (sent[j] == '.' and j > 3 and sent[j - 3:j + 1] in ['U.N.', 'E.U.', 'U.S.']):
                    j += 1
                    continue
                if j != 0:
                    sent = sent[:j].rstrip(' ') + ' ' + sent[j] + ' ' + sent[ j + 1:].lstrip(' ')
                else:
                    sent = sent[j] + ' ' + sent[j + 1:].lstrip(' ')
            else:
                sent = sent[:j].rstrip(' ') + ' ' + sent[j]
            j += 1
        for k in [2, 3]:
            if sent[j:j + k] in ['\'s', 'n\'t']:
                if j < ind:
                    ind += 1
                if j != len(sent) - k:
                    sent = sent[:j].rstrip(' ') + ' ' + sent[j:j + k] + ' ' + sent[ j + k:].lstrip(
                        ' ')
                else:
                    sent = sent[:j].rstrip(' ') + ' ' + sent[j:j + k]
                j += 1
                break
        if sent[j:j + 6] == "cannot":
            if j < ind:
                ind += 1
            sent = sent[:j] + 'can not' + sent[j + 6:]
        if sent[j:j + 3] == "s\' ":
            if j < ind:
                ind += 1
            sent = sent[:j] + 's \' ' + sent[j + 3:]
        if sent[j:j + 2] == '  ':
            if j < ind:
                ind -= 1
            sent = sent[:j] + sent[j + 1:]
            continue
        if sent[j:j + 2] in ['AM', 'PM', 'am', 'pm'] and j > 0 and sent[
            j - 1] in '0123456789':
            if j < ind:
                ind += 1
            sent = sent[:j] + ' ' + sent[j:]
            j += 1
        j += 1
    return sent, ind


def get_tag_brute_force(sent, word):
    """
    In cases where it is difficult to match word indexes just find the first
    occurance of a word in a line and output its POS tag
    :param sent:
    :param word:
    :return:
    """
    sent = ' '.join(sent)
    if word not in sent:
        word = re.sub('[^a-z]', '', word)
        if word in ['nt']:
            return "RB"
        if word in ['euros', 'songkhla', 'opinions', 'obamas', 'madness', 'sv',
                    'life', 'area', 'region', 'partners', 'home', 'jurisdictions',
                    'lebanon', 'theater', 'stuffing', 'time', 'afghanistans',
                    'circle', 'worlds', 'apples', 'sunday']:
            return 'N'
        if word in ['selfmedicating', 'thoseaffected', 'affected', 'illegitimate']:
            return 'J'
        if word in ['ve', 'im', 'theyre', 'is', 'weve', 'were', 'come', 'doesnt',
                    'begins', 'dont']:
            return 'V'
        if word in ['a']:
            return 'DT'
        if word in ['me']:
            return 'P'
        if re.match('.*[a-z].*', word):
            print(word)
        return "BAD" + "-" * 100
    word_id = sent.index(word)
    tag = re.sub(r'.*?_(.*?) .*', r'\1', sent[word_id:]).upper()
    # print(word, tag)
    return getGeneralisedPOS(tag)


def get_tags(text):
    """
    A function that gets a list of lines (Chris paper format)
    and returns a list of tags (for target lines)
    :param text:
    :return:
    """
    return verify_data(text, tag_data(text))
