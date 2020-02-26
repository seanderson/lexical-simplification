# this module contains scripts that allow comparing Kriz's CWI dataset to actual newsela alignments

from glob import glob
from tqdm import tqdm
import numpy as np
import re
import sys
from nltk.corpus import stopwords
from aligner import align
from utils import tokenize, lemmatize

STOPWORDS = set(stopwords.words("english"))
PATH_TO_DATASET = "/home/nlp/wpred/datasets/kriz/Newsela_Complex_Words_Dataset.txt"
PATH_TO_ALIGNED = "/home/nlp/wpred/newsela/articles/aligned/"
ALIGN_DICT = {} # the dictionary to store the results of expensive word-to-word alignments
SEPARATOR = "_ALIGNED_WITH_"
COMPLEX = 2
SIMPLE = 0
UNK = 1
ALIGNMENT_STATS = {"total": 0, "unsuccessful": 0}


def create_key(sentence):
    """
    Compacts a sentence into a single string so that different tokenization of the same sentence
    would map to the same key
    :param sentence: a string of a (possibly tokenized) sentence
    :return:         the key with all extra characters removed, everything lowercased, etc.
    """
    return re.sub("[^a-z]", "", sentence.lower())


def load_aligned_data():
    """
    Load all of alignments into one dictionary for quick access
    :return: a dictionary that maps sentence key to a dictionaty that has the folllowing form
    {"art": article_name,
    "lC": complex level,
    "lS": simple level,
    "complex": complex sentence,
    "simple": simple_sentence}
    """
    aligned_data = {}
    filenames = glob(PATH_TO_ALIGNED + '*.txt')
    print("Indexing aligned data...")
    sys.stdout.flush()
    for filename in tqdm(filenames):
        artS, artC = filename.split(SEPARATOR)  # article names
        lS, lC = int(artS[-5:-4]), int(artC[-5:-4])
        art = artC.split('.')[0]
        with open(filename) as file:
            lines = file.readlines()
        lines = [x.strip('\n') for x in lines if x != '\n']  # strip
        for line in lines:
            _, simple, _, _, complex = line.split('\t')
            sentence_key = create_key(complex)
            if sentence_key not in aligned_data:
                aligned_data[sentence_key] = []
            if simple in [x["simple"] for x in aligned_data[sentence_key]]:
                continue
            aligned_data[sentence_key].append({"art": art,
                                               "lC": lC,
                                               "lS": lS,
                                               "simple": simple,
                                               "complex": complex})
    return aligned_data


def get_index(sent, token, index, pairs):
    """
    Given a sentence tokenized by Kriz, and a token that must be in the same sentence but tokenized
    by Sultan, find the index of that word among tokens in sent
    :param sent:  list of string tokens
    :param token:  a word from the sentence that can be assembled from these tokens
    :param index: alignment index to pass to Sultan aligner
    :param pairs: pairs returned by Sultan's aligner
    :return: index of word in sentence
    """
    if token not in sent:
        return -1
    if sent.count(token) == 1:
        return sent.index(token)

    preceding = 0  # how many instances of this word precede this word in a sentence
    for i in range(len(pairs[0])):
        if pairs[1][i][0] == token and pairs[0][i][0] < index:
            preceding += 1

    lastId = -1
    for i in range(len(sent)):
        if sent[i] == token:
            lastId = i
            if preceding == 0:
                break
            preceding -= 1

    return lastId


def get_alignment_complexity_scores(s0, s1):
    """
    Run Sultan's aligner on two sentences and return the list that for each word in the first
    sentence specifies whether it was changed/simplified (1), kept unchanged (2) or cannot be
    linked to any other word in the sentence (0).
    :param s0: the first sentence as a list of tokens
    :param s1: the second sentence as a string
    :return:   see above
    """
    s0 = [x.lower() for x in s0]
    s1 = s1.lower()

    # check if the alignment has been performed before
    dict_key = " ".join(s0) + SEPARATOR + s1
    if dict_key in ALIGN_DICT:
        return ALIGN_DICT[dict_key]

    result = np.full(len(s0), UNK)
    ALIGNMENT_STATS["total"] += 1

    try:
        # tokenize and lemmatize the sentences
        s0_tok = tokenize(" ".join(s0))
        s1_tok = tokenize(s1)
        s0_lem = lemmatize(s0_tok)
        s1_lem = lemmatize(s1_tok)
        pairs = align(s0_tok, s1_tok)  # pairs of sentences aligned by Sultan's word aligner
    except:
        ALIGN_DICT[dict_key] = result
        ALIGNMENT_STATS["unsuccessful"] += 1
        return result

    # iterate over aligned pairs and feel the result array
    for i in range(len(pairs[0])):
        w0, w1 = pairs[1][i][0].lower(), pairs[1][i][1].lower()
        if w0 in STOPWORDS or w1 in STOPWORDS:  # such an alignment doesn't matter
            continue
        if w0 == w1 or s0_lem.get(w0, 'w0') == s1_lem.get(w1, 'w1'):
            # the alignment is valid but it only indicates that the word was kept as it is
            id = get_index(s0, w0, i, pairs)
            if id == -1:
                continue
            result[id] = SIMPLE
        else:
            id = get_index(s0, w0, i, pairs)
            if id == -1:
                continue
            result[id] = COMPLEX

    ALIGN_DICT[dict_key] = result
    return result


def process_sentence(tokens, token_scores, aligned_data):
    """
    Process a sentence in Kriz's CWI dataset
    :param tokens:       the list of tokens of which the sentence is composed
    :param token_scores: the complexity scores for each token (-1 if complexity score is absent)
    :param aligned_data: dictionary returned by index_aligned_data():
    :return: matrix indexed by alignment_complexity_score, KRIZ_complexity_score
    """
    score_matrix = np.zeros((3, 11), dtype=int)
    sentence_key = create_key("".join(tokens))
    if sentence_key not in aligned_data:
        return score_matrix, False
    # The assertion below will fail because not all complex sentences are aligned by Stajner
    # aligner, but, in theory, all sentences in Kriz dataset are from level 0
    # assert(0 in [alignment["lC"] for alignment in aligned_data[sentence_key]])
    align_scores = np.full(len(token_scores), UNK)
    for alignment in aligned_data[sentence_key]:
        if alignment["simple"] == alignment["complex"]:
            align_scores[np.where(align_scores == UNK)] = SIMPLE
            continue
        tmp_scores = get_alignment_complexity_scores(tokens, alignment["simple"])
        align_scores[np.where(tmp_scores == COMPLEX)] = COMPLEX  # mark complex as complex
        for i in range(len(align_scores)):  # mark simple as simple
            if align_scores[i] == UNK and tmp_scores[i] == SIMPLE:
                align_scores[i] = SIMPLE
    for i in range(len(token_scores)):
        if token_scores[i] != -1:
            score_matrix[align_scores[i], token_scores[i]] += 1
    return score_matrix, True


def load_dataset():
    """
    Loads Kriz's dataset as a list of sentence each of which is represented as a dictionary of
    the following form:     {"tokens":      list of tokens in sentence
                            "token_scores": complexity scores for each token (-1 if absent)}
    :return: See above
    """
    with open(PATH_TO_DATASET) as file:
        lines = file.readlines()

    sentences = []
    word, pos, score, curr_sentence = lines[0].strip('\n').split('\t')
    wrong_scores = []
    if int(score) > 10:
        wrong_scores.append(int(score))
        score = 10
    curr_tokens = curr_sentence.split(' ')
    curr_token_scores = np.full(len(curr_tokens), -1)
    curr_token_scores[int(pos)] = int(score)
    assert(curr_tokens[int(pos)] == word)

    print("Loading Kriz Dataset...")
    sys.stdout.flush()
    for line in tqdm(lines):
        word, pos, score, new_sentence = line.strip('\n').split('\t')
        if int(score) > 10:
            wrong_scores.append(int(score))
            score = 10
        if new_sentence != curr_sentence:  # if new sentence is being processed
            sentences.append({"tokens": curr_tokens, "token_scores": curr_token_scores})
            curr_sentence = new_sentence
            curr_tokens = curr_sentence.split(' ')
            curr_token_scores = np.full(len(curr_tokens), -1)
        assert (curr_tokens[int(pos)] == word)
        curr_token_scores[int(pos)] = int(score)
    sys.stdout.flush()
    print("Wrong scores encountered " + str(len(wrong_scores)) + " times. "
                                                                 "These are " + str(wrong_scores))
    sentences.append({"tokens": curr_tokens, "token_scores": curr_token_scores})
    return sentences


def kriz_dataset_stats(dataset):
    """
    Print some statistics about Kriz dataset
    :param dataset: the list returned by load_dataset()
    :return: None
    """
    sys.stdout.flush()
    score_freq = np.zeros(11, dtype=int)  # frequency of different complexity values
    for sent in dataset:
        for score in sent["token_scores"]:
            if score == -1:
                continue
            if score < 0 or score > 10:
                print("Wrong score encountered: %d" % score)
                continue
            score_freq[score] += 1
    print("Total words annotated: %d, among them %d (%.1f percent) are complex, "
          "i.e. have score of 3 or higher" % (np.sum(score_freq), np.sum(score_freq[3:]),
                                              100 * np.sum(score_freq[3:]) / np.sum(score_freq)))
    print("Distribution of scores in percentages goes as follows:")
    print("\t".join([str(i) for i in range(11)]))
    print("\t".join([str(round(100 * score_freq[i]/np.sum(score_freq), 3)) for i in range(11)]))
    all_scores = np.repeat(np.arange(11), score_freq)
    print("Mean is %.2f, median is %d, standard deviation is %.2f" %
          (all_scores.mean(), np.median(all_scores), all_scores.std()))


def process_dataset():
    """
    Process all of Kriz's dataset sentence by sentence
    :return:
    """
    dataset = load_dataset()
    kriz_dataset_stats(dataset)
    aligned_data = load_aligned_data()
    score_matrix = np.zeros((3, 11), dtype=int)
    aligned_total = 0
    print("Processing Kriz dataset...")
    sys.stdout.flush()
    for sentence in tqdm(dataset):
        curr_matrix, aligned = process_sentence(sentence["tokens"],
                                                     sentence["token_scores"],
                                                     aligned_data)
        aligned_total += aligned
        score_matrix += curr_matrix

    sys.stdout.flush()
    print("%d sentences total, %d aligned (%2.1f percent)." %
          (len(dataset), aligned_total, 100 * aligned_total / len(dataset)))
    print("Word aligner failed to extract word-to-word alignment\n"
          "for %.2f percent of unidentical sentence alignments" %
          (100 * ALIGNMENT_STATS["unsuccessful"] / ALIGNMENT_STATS["total"]))

    print("Kriz complexity score distribution for different alignment situations:")
    print("               " + "\t".join([str(i) for i in range(11)]))
    print("Word kept      " + "\t".join([str(round(100 * score_matrix[SIMPLE][i]/np.sum(score_matrix[SIMPLE]), 1))
    for i in range(11)]))
    print("Word removed:  " + "\t".join([str(round(100 * score_matrix[UNK][i]/np.sum(score_matrix[UNK]), 1))
    for i in range(11)]))
    print("Word replaced: " + "\t".join([str(round(100 * score_matrix[COMPLEX][i]/np.sum(score_matrix[COMPLEX]), 1))
    for i in range(11)]))

    print("Word kept total: %d, word removed total: %d, word replaced total: %d" % (
        np.sum(score_matrix[SIMPLE]), np.sum(score_matrix[UNK]), np.sum(score_matrix[COMPLEX])))

    ts = np.repeat(np.arange(11), score_matrix[SIMPLE, :])
    print("If word is simple when it is kept, scores for simple words have:")
    print("the mean of %.2f, the median of %d, standard deviation of %.2f." %
          (ts.mean(), np.median(ts), ts.std()))
    tc = np.repeat(np.arange(11), score_matrix[COMPLEX, :])
    print(
        "If word is complex when it is replaced with another word, scores for complex words have:")
    print("the mean of %.2f, the median of %d, standard deviation of %.2f." %
          (tc.mean(), np.median(tc), tc.std()))
    tu = np.repeat(np.arange(11), score_matrix[UNK, :])
    print("For other words, the scores have:")
    print("the mean of %.2f, the median of %d, standard deviation of %.2f." %
          (tu.mean(), np.median(tu), tu.std()))


if __name__ == "__main__":
    process_dataset()