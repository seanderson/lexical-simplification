# extracts word-to-word alignments from sentence-to-sentence alignments


from aligner import *
from utils import *
from nltk.corpus import stopwords
from glob import glob


alignment_dict = {"", }


def process_sentence(s0, s1):
    """
    Get two sentences (strings) that were aligned together and extract
    word-to-word alignment info from it
    :param s0:
    :param s1:
    :return: list of pairs of aligned words that are not identical
    """
    s0_tok = tokenize(s0)
    s1_tok = tokenize(s1)
    s0_lem = lemmatize(s0_tok)
    s1_lem = lemmatize(s1_tok)
    try:
        pairs = align(s0_tok, s1_tok)[1]  # pairs of aligned words
    except:
        print("Unicode Decode Error!")
        return ""
    different_pairs = []  # list of aligned words that are different
    for pair in pairs:
        if pair[0].lower() == pair[1].lower() or s0_lem[pair[0].lower()] == s1_lem[pair[1].lower()]:
            continue
        if pair[0].lower() in set(stopwords.words("english")):
            continue
        if pair[1].lower() in set(stopwords.words("english")):
            continue
        different_pairs.append(": ".join(pair))
    return ", ".join(different_pairs)


def process_file(filename):
    """
    Process an alignemnt file
    :param filename:
    :return:
    """
    with open(filename) as file:
        alignments_1_to_1 = [line.strip('\n').split('\t') for line in file.readlines() if line != '\n']

    # creating 1-to-n alignments
    alignments = {}
    for alignment in alignments_1_to_1:
        complex = alignment[4]
        simple = alignment[1]
        if complex not in alignments:
            alignments[complex] = simple
        else:
            alignments[complex] += " " + simple

    for complex in alignments:
        simple = alignments[complex]
        key = re.sub("[^a-z]", "", complex.lower()) + re.sub("[^a-z]", "", simple.lower())
        if key in alignment_dict:
            continue
        alignment_dict.add(key)
        pairs = process_sentence(complex.split, simple.split)
        if pairs:
            print(complex)
            print(simple)
            print(pairs + '\n')


if __name__ == "__main__":
    filenames = glob('/home/nlp/wpred/newsela/articles/aligned/*.txt')
    for filename in filenames:
        print("\n\n\n" + filename)
        process_file(filename)