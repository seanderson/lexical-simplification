"""
Combine probabilities of predicted words (nnet or ngram) with word
changes from aligned sentences.
"""

from alignutils import *
import newselautil as nsla
from prepData import *
import utilsKeras as utils
import h5py
import pylab as pl
import classpaths as path

DEBUG = False

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
    articles = nsla.loadMetafile()
    simple = []
    complex = []
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
                                                  nn_representation_of_sentence)
                data = [data[k] + tmp_data[k] for k in range(len(data))]
                simple += data[1] + data[3]
                complex += data[0] + data[2]
                curr_al += 1
    return data


def analyze(aligned_output, nn_output, nn_representation_of_sentence):
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
    for i in range(len(nn_representation_of_sentence)):
        word = nn_representation_of_sentence[i]
        if word == PAR_START or word == SENT_START or word == SENT_END:
            offset += 1
        elif aligned_output[i - offset][0] == '_':
            # word is complex
            if nn_output[i][0][1] == nn_output[i][1][1]:
                complex_correct.append(nn_output[i][1][0])
            else:
                complex_wrong.append(nn_output[i][1][0])
        else:
            # word is not complex
            if nn_output[i][0][1] == nn_output[i][1][1]:
                simple_correct.append(nn_output[i][1][0])
            else:
                simple_wrong.append(nn_output[i][1][0])
    data = [complex_correct,simple_correct,complex_wrong,simple_wrong]
    return data


def main(probsFile, snum=-1):
    """
    Get various statistics for the probability file
    :param probsfile:
    :param snum: If snum!=-1, only the first snum sentences will be considered
    :return:
    """
    data = sentence_data(path.nnetFile, path.indexFile, probsFile, snum)
    # print([len(x) for x in data])
    # print(data[1])
    print("Plotting simple_correct")
    pl.hist([x * 100 for x in data[1]], bins=range(0, 101, 1))
    pl.show()
    print("Plotting complex_correct")
    # print(data[0])
    pl.hist([x * 100 for x in data[0]], bins=range(0, 101, 1))
    pl.show()
    print("Plotting simple_wrong")
    # print(data[3])
    pl.hist([x * 100 for x in data[3]], bins=range(0, 101, 1))
    pl.show()
    print("Plotting complex_wrong")
    # print(data[2])
    pl.hist([x * 100 for x in data[2]], bins=range(0, 101, 1))
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
