"""
List of utilities that can be used on the databases before or after tagging them
"""

import sys
import re

KAUCHAK_DIRECTORY = "/home/nlp/wpred/text_databases/KW_splitted/"
DEBUG = False  # if True, print the vocabulary in a way that is easy to read
# (but less easy to process with a computer)
TOTAL = "Total"
# A string that is used as a dictionary key for storing the total count
MAX_WORD_LENGTH = 25


def scan_file(filename, databasename, voc):
    """
    Add the words in this file to the vocabulary
    :param filename: absolute path to a file
    :param databasename: the name of the database this file belongs to
    :param voc: the vocabulary to append the word to
    :return:
    """
    with open(filename) as database:
        for line in database:
            for word in line.rstrip('\n').split(' '):
                if word != "":
                    if word not in voc:
                        voc[word] = {}
                    if TOTAL not in voc[word]:
                        voc[word][TOTAL] = 0
                    if databasename not in voc[word]:
                        voc[word][databasename] = 0
                    voc[word][TOTAL] += 1
                    voc[word][databasename] += 1


def print_vocabulary(filename, voc, entries):
    """
    Print out the vocabulary
    :param filename: the name of the file to print the vocabulary to
    :param voc: the vocabulary to print out
    :param entries: a list the first element of which is the TOTAL variable and
    all the others are the possible database names
    :return:
    """
    sorted_keys = sorted(voc.keys())
    with open(filename, 'w') as voc_file:
        begin_tag = "Word"
        if DEBUG:
            begin_tag += ' ' * (MAX_WORD_LENGTH - len(begin_tag))
        voc_file.write('\t'.join([begin_tag] + entries) + '\n')
        for word in sorted_keys:
            if DEBUG and len(word) < MAX_WORD_LENGTH:
                line = word + " " * (MAX_WORD_LENGTH - len(word)) + "\t"
            else:
                line = word + "\t"
            for entry in entries:
                if entry not in voc[word]:
                    line += "0\t"
                else:
                    line += str(voc[word][entry]) + "\t"
            voc_file.write(line + "\n")


def lowercase_voc(voc_name):
    voc = {}
    with open(voc_name) as voc_file:
        first_line = True
        for line in voc_file:
            if first_line:
                entries = line.rstrip('\n').split('\t')[1:]
                first_line = False
                continue
            line = line.rstrip('\t\n').split('\t')
            word = line[0].split('_')
            word = '_'.join(word[:-1]).lower() + '_' + word[-1]
            if word not in voc:
                voc[word] = {}
                for i in range(len(line) - 1):
                    voc[word][entries[i]] = int(line[i + 1])
            else:
                for i in range(len(line) - 1):
                    voc[word][entries[i]] += int(line[i + 1])
    print_vocabulary(voc_name, voc, entries)


def process_vocabulary(voc_name, new_voc_name, min_freq):
    """
    Create a version of a vocabulary by removing all the low-freqency words
    :param voc_name:
    :param new_voc_name:
    :param min_freq: minimal acceptable word-count
    :return:
    """
    with open(new_voc_name, 'w') as new_voc_file:
        with open(voc_name) as voc_file:
            first_line = True
            for line in voc_file:
                if first_line:
                    first_line = False
                    continue
                if int(line.split('\t')[1]) >= min_freq:
                    new_voc_file.write(line)


def preprocess_kauchak(name):
    """
    Remove the article titles and paragraph number from the Kauchak data
    :param name:
    :return:
    """
    with open(KAUCHAK_DIRECTORY + name) as to_read:
        lines = to_read.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].split('\t')[-1].rstrip('\n')
    with open(KAUCHAK_DIRECTORY + name, 'w') as to_write:
        to_write.write('\n'.join(lines) + '\n')


def split_kauchak(name):
    """
    Another way to preprocess teh Kauchak Wikipedia that just splits the
    original files by article title
    :param name:
    :return:
    """
    with open(KAUCHAK_DIRECTORY + name) as to_read:
        lines = to_read.readlines()
    i = 0
    title, _, sent = lines[0].rstrip('\n').split('\t')
    while i < len(lines):
        curr_title = title  # the title of the article currently processed
        curr_sents = []  # the list of sentences from this article
        while curr_title == title:
            curr_sents.append(sent)
            i += 1
            if i == len(lines):
                break
            title, _, sent = lines[i].rstrip('\n').split('\t')
        curr_title = re.sub('[^a-zA-Z]', '_', curr_title)
        with open(KAUCHAK_DIRECTORY +
                  name.split('.')[0] + "_" + curr_title + ".txt",
                  'w') as newfile:
            newfile.write('\n'.join(curr_sents))


def clean(filename):
    """
    Remove anything inside the angle brackets and any line that has no letters
    in it
    :param filename:
    :return:
    """
    print("Processing " + filename)
    with open(filename + ".cleaned", 'w') as output:
        with open(filename) as input_file:
            for sent in input_file:
                sent = sent.rstrip('\n').split(' ')
                do_not_append = False
                newsent = []
                for word in sent:
                    if '<' in word:
                        do_not_append = True
                    elif '>' in word:
                        do_not_append = False
                    elif not do_not_append:
                        newsent.append(word)
                newsent = ' '.join(newsent) + '\n'
                if re.match('.*[a-zA-Z].*', newsent):
                    output.write(newsent)


if __name__ == "__main__":
    # Preprocessing Kauchak database:

    # preprocess_kauchak("simple.txt")
    # preprocess_kauchak("normal.txt")

    # Another way to process the Kauchak database:

    # split_kauchak("simple.txt")
    # split_kauchak("normal.txt")

    # Cleaning the files after tagging:

    # for line in sys.stdin:
        # clean(line.rstrip('\n'))

    # Building a vocabulary:

    """
    if len(sys.argv) < 3:
        print("Please provide the name of the metafile\n" +
              "followed by the name of the file to print the vocabulary to")
        exit(-1)
    databases = [TOTAL]
    voc = {}
    with open(sys.argv[1]) as metafile:
        for line in metafile:
            databasename = line.rstrip('\n').split('/')[-2]
            print("Processing " + databasename + " database...")
            databases.append(databasename)
            with open(line.rstrip('\n')) as database_metafile:
                for tagged_file in database_metafile:
                    scan_file(tagged_file.rstrip('\n'), databasename, voc)
    print_vocabulary(sys.argv[2], voc, databases)"""

    # freq = 20
    lowercase_voc("/home/nlp/wpred/text_databases/voc_uppercase.txt")
    # process_vocabulary("/home/nlp/wpred/text_databases/voc.txt",
    #                    "/home/nlp/wpred/text_databases/voc_freq>="+str(freq)+".txt", freq)
