"""
This script is used by process.sh
"""

import sys

S_HEADER = "Sentence #"
# this is how coreNLP tokenizer introduces a new sentence
W_HEADER = "[Text="
# this is how coreNLP tokenizer introduces a new word


def compact(filename):
    """
    Reduce (tenfold) the size of the coreNLP output file by including the
    tags directly in the text (each word would be followed by an '_' and then
    by a tag)
    :param filename: A file to process
    :return:
    """
    first_line = True  # True, if the first line was not processed yet
    with open(filename) as tagged_file:
        lines = tagged_file.readlines()
    output = ""
    curr = []  # current line
    for sent in lines:
        if len(sent) > len(S_HEADER) and sent[:len(S_HEADER)] == S_HEADER:
            if first_line:
                first_line = False
            else:
                output += ' '.join(curr) + "\n"
                curr = []
        elif len(sent) > len(W_HEADER) and sent[:len(W_HEADER)] == W_HEADER:
            sent = sent.split(' ')
            word = sent[0][len(W_HEADER):]
            tag = sent[-1].split('=')[-1][:-2]
            curr.append(word + "_" + tag)
    output += ' '.join(curr) + "\n"

    with open(filename, 'w') as tagged_file:
        tagged_file.write(output)


if __name__ == "__main__":
    for line in sys.stdin:
        compact(line.rstrip('\n'))
