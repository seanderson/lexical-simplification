"""
Combine probabilities of predicted words (nnet or ngram) with word
changes from aligned sentences.
"""

from alignutils import *
from newselautil import *
from prepData import *
import utilsKeras as utils
import h5py
import pylab as pl
import classpaths as path

DEBUG = False
count_total = 0
count = 0


def sentence_data(nnetfile, indexfile, probsfile, snum=-1):
    """
    Get alignment output and match it with the output of the neural networks.
    Return two lists of probability values. One for words that are not
    simplified in the newsela data and the other one for words that are.
    :param nnetfile:    among other things, contains the vocabulary that can be
                        used to map indexes to words
    :param indexfile:   contains the information about articles' slugs,
                        number of paragraphs in them and number of sentences
                        in each paragraph
    :param probsfile:   the file with probabilities assigned to words by the
                        neural network. probsfile[i][j] contains the data for
                        the j-th word of the i-th sentence. probsfile[i][j][k]
                        is a tuple of two elements. The first is the probability
                        and the second is the word index. k=0 corresponds to the
                        actual correct word, k=1,2,3 - to top3 predictions made
                        by the network
    :param snum:        if snum!=-1, only the first snum sentences will be
                        considered
    :return:            two lists of probability values
                        (simple, complex)
    """
    h5fd = h5py.File(probsfile, 'r')
    data = [[], [], [], []]
    with bz2.BZ2File(nnetfile, 'r') as handle:
        (invoc, sentences) = pickle.load(handle)
    abs_id = -1  # absolute id of a sentence within the corpus
    filenames, npars, sindlst = read_index_file(indexfile)
    articles = loadMetafile()
    simple = []
    complex = []
    word_freq = []
    predictions = utils.readProbs(h5fd)
    for i, art in enumerate(filenames[:]):
        slug, lang, lvl = art.split('.')
        alignments = get_aligned_sentences(articles, slug, 0, 1)
        if DEBUG:
            print (slug + '-' * 80)
        curr_al = 0  # the index of the next alignment to consider
        curr_id = -1  # id of the sentence within the current article

        for ip in range(npars[i]):  # each par for art
            for js in range(sindlst[i][ip]):  # sentences for each par
                abs_id += 1
                curr_id += 1
                if abs_id == snum:
                    return simple, complex
                if len(predictions) <= abs_id:
                    print "Error: Ran out of probabilities in sentenceData"
                    sys.exit(-1)

                sent_begin_with_par = [invoc[y] for y in sentences[abs_id]][0] == PAR_START
                nn_output = predictions[abs_id]
                nn_representation_of_sentence = [invoc[int(x[0][1] - 1)] for x in nn_output]

                if DEBUG:
                    print('NN sent:' + ' '.join(nn_representation_of_sentence))
                if curr_al >= len(alignments) or \
                        alignments[curr_al].ind0 != curr_id:
                    if DEBUG:
                        print("No corresponding alignment\n")
                    continue
                alignment = alignments[curr_al].mark_simplified()
                if DEBUG:
                    print('Aligned sent:' + ' '.join(alignment) + "\n")

                aligned_len = len(alignment)
                nn_len = len(nn_representation_of_sentence)
                if (sent_begin_with_par and nn_len != aligned_len + 2) or \
                        (not sent_begin_with_par and nn_len != aligned_len + 1):
                    print "Error: Lengths are not equal"
                    print('NN sent:' + ' '.join(nn_representation_of_sentence))
                    print('Aligned sent:' + ' '.join(alignment) + "\n")
                    print(str(nn_len) + " : " + str(aligned_len))
                    continue  # TODO: solve the problem with apostrophes
                    # sys.exit(-1)
                tmp_data = analyze(alignment, nn_output,
                                                  nn_representation_of_sentence, invoc)
                data = [data[k] + tmp_data[k] for k in range(len(data))]
                simple += data[1] + data[3]
                complex += data[0] + data[2]
                curr_al += 1
                word_freq = find_freq_given_prob(nn_output, nn_representation_of_sentence, word_freq, prob=157)
    print word_freq
    return data


def analyze(aligned_output, nn_output, nn_representation_of_sentence, voc, topN=1):
    """
    Given that aligned output and nn_output represent the same sentence,
    return the probabilities that the NN assigns to complex and simplified words
    :param aligned_output:
    :param nn_output:
    :param nn_representation_of_sentence:
    :return:
    """
    offset = 0
    complex_correct = []
    simple_correct = []
    complex_wrong = []
    simple_wrong = []
    # TODO: ALL of the unknown tags should be ignored
    for i in range(len(nn_representation_of_sentence)):
        word = nn_representation_of_sentence[i]
        append_value = sum([x[0] for x in nn_output[i][1:topN+1]])

        if voc[int(nn_output[i][0][1])-1] == "@UNK" and voc[int(nn_output[i][0][1])-1] == "@NUM":
            continue
        global count
        if voc[int(nn_output[i][0][1])-1] == u'``':
            count += 1
        # global count
        # global count_total
        # if nn_output[i][1][0] < .158 and nn_output[i][0][1] > 0.156:
            # if voc[int(nn_output[i][0][1]) - 1] in ['the', 'and', 'The', 'And', 'it', 'It']:
                # count += 1
            # count_total += 1

        if topN != 1:
            lemmas_equal = lemmas_are_equal(nn_output[i][0][1],
                                        nn_output[i][1:topN + 1][1], voc)
        else:
            lemmas_equal = lemmas_are_equal(nn_output[i][0][1],
                                            [nn_output[i][1][1]], voc)
        if word == PAR_START or word == SENT_START or word == SENT_END:
            offset += 1
        elif aligned_output[i - offset][0] == '_':
            # word is complex
            if lemmas_equal:
                complex_correct.append(append_value)
            else:
                complex_wrong.append(append_value)
        else:
            # word is not complex
            if lemmas_equal:
                simple_correct.append(append_value)
            else:
                simple_wrong.append(append_value)
    data = [complex_correct,simple_correct,complex_wrong,simple_wrong]
    return data


def find_freq_given_prob(nn_output, nn_representation_of_sentence, output_list = [[]], prob=157, topN=1):
    """
    finds the number of occurences of words in a sentence at a specific
    probability
    :param nn_output:
    :param nn_representation_of_sentence:
    :param output_list: list to output data to
    :param prob: probability to search for (decmal times 1000, % times 10)
    :param topN: how many guesses of the neural network to consider
    :return: a list with [[word][# of occurrences total]]
    """
    for i in range(len(nn_representation_of_sentence)):
        word = nn_representation_of_sentence[i]
        append_value = sum([x[0] for x in nn_output[i][1:topN + 1]])

        if int(append_value * 1000) == prob:
            has_word = False
            for x in range(len(output_list)):
                if len(output_list) != 0 and word == output_list[x][0]:
                    has_word = True
                    output_list[x][1] += 1;
                    break

            if not has_word:
                output_list.append([word, 1])
    return sorted(output_list, key=lambda x: x[1], reverse=True)


def lemmas_are_equal(ind0, indexes, voc):
    """
    :param ind0: the index of the first lemma
    :param indexes: the indexes of the other lemmas
    :param voc: vocabulary
    :return:
    """
    ind0 = int(ind0)
    result = False
    for ind1 in indexes:
        ind1 = int(ind1)
        lemma0 = Lemmatizer.lemmatize(voc[ind0 - 1])
        lemma1 = Lemmatizer.lemmatize(voc[ind1 - 1])
        if lemma0 != 'a':
            result = result or (lemma0 == lemma1 or ind0 == ind1)
        else:
            result = result or (ind0 == ind1)
    return result


def get_frequency(data):
    """
    returns a count of how many words occur for a specific probability
    :param data:
    :return: the count as four arrays(corresponding to data type) of 1001
    indexes(corresponding to probability)
    """
    hist_type_data = numpy.zeros((4, 1001), int)
    for category in range(len(data)):
        for word in data[category]:
            bin = int(1000*word)
            hist_type_data[category][bin] += 1
    return hist_type_data


def find_spikes(frequencies, breakpoint):
    """
    finds occurrences of probabilities who have a greater numebr of words than
    the breakpoint
    :param frequencies: list to read from. Generated by get_frequency()
    :param breakpoint:
    :return: a list of the spikes greater than the breakpoint in format
    [(probability, number of occurrences)]
    """
    spikes = [()]
    for category in range(len(frequencies)):
        for prob in range(len(frequencies[category])):
            if frequencies[category][prob] > breakpoint:
                spikes.append((prob, frequencies[category][prob]))
    return spikes


def main(probsFile, snum=-1):
    """
    Get various statistics for the probability file
    :param probsfile:
    :param snum: If snum!=-1, only the first snum sentences will be considered
    :return:
    """
    data = sentence_data(path.nnetFile, path.indexFile, probsFile, snum)
    global count_total
    global count
    # print("Fraction is " + str(round(float(count / count_total), 3)))
    print("Count = " + str(count))
    spikes = find_spikes(get_frequency(data), 2500)
    print(spikes)
    print("Complex: correct " + str(round(float(len(data[0]))/(len(data[0]) + len(data[2])) * 100, 3)) + "% of times")
    print("Simple: correct " + str(
        round(float(len(data[1]))/(len(data[1]) + len(data[3])) * 100, 3)) + "% of times")
    # print(data[1])
    print("Plotting simple_correct")
    pl.hist([x * 1000 for x in data[1]], bins=range(0, 1001, 1))
    pl.show()
    print("Plotting complex_correct")
    # print(data[0])
    pl.hist([x * 1000 for x in data[0]], bins=range(0, 1001, 1))
    pl.show()
    print("Plotting simple_wrong")
    # print(data[3])
    pl.hist([x * 1000 for x in data[3]], bins=range(0, 1001, 1))
    pl.show()
    print("Plotting complex_wrong")
    # print(data[2])
    pl.hist([x * 1000 for x in data[2]], bins=range(0, 1001, 1))
    pl.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("The default configuration will be run. For non-default "
              "configurations, use:")
        print("python simpProbabilities.py [probability-file-name] "
              "[number-of-sentences-to-look-at]")
        main(path.DEFAULT_MODEL_NAME)
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main(sys.argv[1], sys.argv[2])
