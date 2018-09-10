"""
This modules contains utils for working with already aligned articles

def get_aligned_sentences(metafile, slug, level1, level2, auto=True): Return aligned sentences..
"""

from src.newselautil import *
from src import classpaths as path
import numpy
import re
# import utils_for_reno_kriz_data
import sys
sys.path.insert(0, '/home/nlp/newsela/monolingual-word-aligner/monolingual-word-aligner')

# import aligner


class Alignment(object):

    """ a class that represents an alignment """
    FILL = "@IGNORE"

    def __init__(self, sent0, ind0, p_ind0, s_ind0, part0,
                 sent1, ind1, p_ind1, s_ind1, part1):
        """
        All indexes are zero-based.
        :param sent0: the sentence from the first articles
        :param ind0: the absolute index of this sentence in the file
        :param p_ind0: the index of the paragraph this sentence appears in
        :param s_ind0: the index of this sentence relative to the beginning of
        the paragraph. getTokParagraphs(...)[...][p_ind0][s_ind0] will return
        sent0
        :param part0: the part of the sentence that was aligned.
        Parts are separated by semicolons, i.e. sent0.split(';')[part0] is what
        was actually aligned by the algorithm
        :param sent1: same for the second article
        :param ind1:[idx[0]
        :param p_ind1:
        :param s_ind1:
        :param part1:
        """
        self.sent0 = sent0.rstrip(' ')
        self.sent1 = sent1.rstrip(' ')
        # self.sent0 = sent0.split(';')[part0]
        # self.sent1 = sent1.split(';')[part1]
        self.part0 = part0
        self.part1 = part1
        self.ind0 = ind0
        self.ind1 = ind1
        self.s_ind0 = s_ind0
        self.s_ind1 = s_ind1
        self.p_ind0 = p_ind0
        self.p_ind1 = p_ind1

    def mark_simplified(self, no_stopwords=True, additionalTokenizer = True):
        """
        Analyze the alignment and compare the original sentence with the 
        simplified version
        :return: a list of tokens (words) that together form the original
                 sentence. Tokens surrounded with '_' signs were simplified
                 by the authors of newsela corpus 
        """
        if additionalTokenizer:
            simple_sentence = [x.lower() for x in tokenize(self.sent1)]
            complex_sentence = [x.lower() for x in tokenize(self.sent0)]
        else:
            simple_sentence = [x.lower() for x in self.sent1.split(' ')]
            complex_sentence = [x.lower() for x in self.sent0.split(' ')]
        new_s = []
        for lw in complex_sentence:
            if lw == '':
                new_s.append(Alignment.FILL)
            # elif lw not in simple_sentence and (
                    # not no_stopwords or lw not in STOPWORDS):
            elif lw not in simple_sentence and lw not in STOPWORDS:
                # not found, which means that it probably was simplified
                new_s.append('_' + lw + '_')
            else:
                new_s.append(lw)
        return new_s


def get_lowest_element_with_slug(slug, metafile):
    """
    return the position of the first element with the given slug within the metafile. Performs the binary search
    :param slug: the slug to search for
    :param metafile: the metafile to use
    :return: The position of the first element with this slug
    """
    hi = len(metafile) - 1  # search for slug
    lo = 0
    while lo < hi:
        mid = int((hi + lo) / 2)
        if metafile[mid]['slug'] < slug:
            lo = mid + 1
        else:
            hi = mid
    if metafile[lo]['slug'] != slug:
        print("No such slug: " + slug)
        return
    # ASSERT: lo contains the slug
    while (lo > 0) and (
        metafile[lo]['slug'] == metafile[lo - 1]['slug']):  # lo should point to the first article with slug
        lo -= 1
    return lo


def replace(threeDimArray, old, new):
    """
    Get a two dimensional array and replace all the old values with the new ones
    :param threeDimArray: the array to interate over
    :param old: the old value
    :param new: the new value
    :return: None
    """
    for twoDim in threeDimArray:
        for oneDim in twoDim:
            for i in range(len(oneDim)):
                if oneDim[i] == old:
                    oneDim[i] == new


def get_aligned_sentences(metafile, slug, level1, level2, auto=True, use_spacy=False):
    """
    Returns the list of Alignment objects sorted by the absolute index.
    :param metafile:        the metafile loaded with newselautils.loadMetafile()
    :param slug:            the slug of the aligned articles
    :param level1:          the lower level of the alignment
    :param level2:          the upper level of the alignment
    :param auto:            true if alignments made by the algorithm are to be
                            loaded, false otherwise (for manual alignemnets)
    :return:
    """
    if level1 >= level2:
        print("level1 is greater than level2")
        return
    lo = get_lowest_element_with_slug(slug, metafile)
    if lo is None:
        return
    allParagraphs = [getTokParagraphs(metafile[lo + level1], False, False),
                     getTokParagraphs(metafile[lo + level2], False, False)]
    result = []

    for article in allParagraphs:
        for j in range(len(article)):
            paragraph = article[j]
            if j == 0:
                paragraph[0] = (paragraph[0], 0, 0)
            else:
                paragraph[0] = (paragraph[0], 0, article[j-1][-1][2] + 1)
            for i in range(len(paragraph)-1):
                paragraph[i+1] = (paragraph[i+1], paragraph[i][1] + len(paragraph[i][0].split(";")),
                                  paragraph[i][2] + 1)

    sentCount = ([], [])
    for i in range(len(allParagraphs)):
        for j in range(len(allParagraphs[i])):
            sentCount[i].append(numpy.ndarray(len(allParagraphs[i][j]),numpy.int8))
            sentCount[i][j].fill(-1)
    # sentCount[0][i][j] is the block of alignment in which the i-th sentence of the j-th paragraoh of the first article
    #  appears. sentCount[0][i][j] is the same thing for the second article
    # if the same sentence appears in two blocks of alignment, the blocks are concatenated

    if auto:
        directory = path.OUTDIR_SENTENCES
        i = 3
    else:
        directory = path.MANUAL_SENTENCES
        i = 1

    with open(directory + slug+"-cmp-"+str(level1)+"-"+str(level2)+".csv") as file:
        f = file.readlines()
        while i < len(f):
            line = f[i].split("\t")
            current = []
            blockId = len(result)  # current is added to result[oldBlock]
            for alignment in line:
                alignment = alignment.split(",")
                first = convert_coordinates(list(map(int, re.findall(r'\d+', alignment[0]))), allParagraphs[0])
                second = convert_coordinates(list(map(int, re.findall(r'\d+', alignment[1]))), allParagraphs[1])

                if blockId == len(result):
                    if (sentCount[0][first[0]][first[1]] != -1)and(sentCount[0][first[0]][first[1]] != len(result)):
                        blockId = sentCount[0][first[0]][first[1]]
                        replace(sentCount, len(result), blockId)
                    elif (sentCount[1][second[0]][second[1]] != -1)and(sentCount[1][second[0]][second[1]]!=len(result)):
                        blockId = sentCount[1][second[0]][second[1]]
                        replace(sentCount, len(result), blockId)
                if sentCount[0][first[0]][first[1]] == -1:
                    sentCount[0][first[0]][first[1]] = blockId
                elif sentCount[0][first[0]][first[1]] != blockId:
                    current += result[sentCount[0][first[0]][first[1]]]
                    result[sentCount[0][first[0]][first[1]]] = None
                    replace(sentCount, sentCount[0][first[0]][first[1]], blockId)
                if sentCount[1][second[0]][second[1]] == -1:
                    sentCount[1][second[0]][second[1]] = blockId
                elif sentCount[1][second[0]][second[1]] != blockId:
                    current += result[sentCount[1][second[0]][second[1]]]
                    result[sentCount[1][second[0]][second[1]]] = None
                    replace(sentCount, sentCount[1][second[0]][second[1]], blockId)

                ind0 = allParagraphs[0][first[0]][first[1]][2]
                ind1 = allParagraphs[1][second[0]][second[1]][2]
                sent0 = allParagraphs[0][first[0]][first[1]][0]
                sent1 = allParagraphs[1][second[0]][second[1]][0]
                current.append(Alignment(sent0, ind0, first[0], first[1], first[2],
                                         sent1, ind1, second[0], second[1], second[2]))
            if blockId == len(result):
                result.append(current)
            else:
                result[blockId] += current
            i += 1
        i = 0
        while i < len(result):
            if result[i] is None:
                del result[i]
            else:
                i += 1
    # result accounts for N-1, N-N and 1-N alignments. new_result does not
    new_result = []
    for block in result:
        if len(block) == 1:
            if len(block[0].sent0.split(';')) == 1:
                new_result += block
        else:
            # TODO: this is a preliminary way to fix things, not the most
            # elegant one
            block = sorted(block, key=lambda smth: smth.ind0, reverse=False)
            i = 1
            count = 0
            while i < len(block):
                if block[i].ind0 == block[i-1].ind0:
                    block[i-1].sent1 += " " + block[i].sent1
                    count += 1
                    del block[i]
                else:
                    if count < len(block[i-1].sent0.split(';')):
                        del block[i-1]
                    else:
                        i += 1
                    count = 0
            if count < len(block[-1].sent0.split(';')):
                del block[-1]
            new_result += block
    if use_spacy:
        with io.open(path.BASEDIR + '/articles/' + slug + '.en.0.txt.spacy') as file:
            lines = file.readlines()
        for alignment in new_result:
            alignment.sent0 = lines[alignment.ind0]
    return sorted(new_result, key=lambda smth: smth.ind0, reverse=False)


def convert_coordinates(old, pars):
    """
    Convert coordinates from those written in the -cmp- files to those needed in alignutils. First of all, the
    coordinates are made zer0-based instead of 1-based. Secondly, the part of the sentence separated by a semicolon
    are no longer treated as separated sentences
    :param old: old coordinates (n_of_paragraph, n_of_phrase)
    :param pars: the paragraphs for the article for which the coordinates are needed
    :return: new coordinates (n_of_paragraph, n_of_sentence, n_of_phrase)
    """
    old = (old[0]-1, old[1]-1)
    i = 0
    while (i < len(pars[old[0]])-1)and(old[1] >= pars[old[0]][i+1][1]):
        i += 1
    return old[0], i, old[1]-pars[old[0]][i][1]


def concatenate(alignments):
    i = 0
    while i < len(alignments):
        j = i + 1
        while j < len(alignments):
            if alignments[i][0][1][0] == alignments[j][0][1][0]:
                alignments[i][1][0] += " " + alignments[j][1][0]
                alignments[i][1][1] += alignments[j][1][1]
                # print("Created: " + alignments[i][1][0])
                del alignments[j]
            else:
                j += 1
        i += 1
    i = 0
    while i < len(alignments):
        j = i + 1
        while j < len(alignments):
            if not set(alignments[i][1][1]).isdisjoint(alignments[j][1][1]):
                alignments[i][0][0] += " " + alignments[j][0][0]
                alignments[i][0][1] += alignments[j][0][1]
                # print("Created(2): " + alignments[i][0][0])
                del alignments[j]
            else:
                j += 1
        i += 1


def sultan_aligner(sent0, sent1, tags0, tags1, expectation):
    """

    :param sent0:
    :param sent1:
    :return:
    """
    if len(sent0) != len(sent1):
        return
    indexes, alignments = aligner.align(sent0, sent1)
    alignments = [([alignments[i][0], [indexes[i][0] - 1]],
                   [alignments[i][1], [indexes[i][1] - 1]]) for i in range(len(alignments))]
    concatenate(alignments)
    alignments = [x for x in alignments if x[0][0].lower() != x[1][0].lower() and x[0][0].lower() not in STOPWORDS and x[1][0].lower() not in STOPWORDS]
    i = 0
    while i < len(alignments):
        a = alignments[i]
        if len(a[0][1]) == 1 and len(a[1][1]) == 1:
            alignments[i] = ([a[0][0], tags0[a[0][1][0]]], [a[1][0], tags1[a[1][1][0]]])
            if alignments[i][0][1][0] != alignments[i][1][1][0]:
                # print("parts of speech are different: " + str(alignments[i]))
                pass
            if smart_lemmatize(alignments[i][0][0], alignments[i][0][1]) == smart_lemmatize(alignments[i][1][0], alignments[i][1][1]):
                del alignments[i]
                continue
        i += 1
    if len(alignments) == expectation:
        print(' '.join(sent0))
        print (' '.join(sent1))
        print(str(alignments) + '\n')


def output_alignments(file, sentpairs, slug, all_alignments, debug=False):
    """

    :param file:
    :param sentpairs:
    :return:
    """
    ARTICLES = ['a', 'an', 'the']
    als = []
    for alignment in sentpairs:
        if alignment.sent0 == alignment.sent1:
            continue
        alignment.sent0 = re.sub('  ', ' ', alignment.sent0).rstrip(' ').lstrip(' ')
        alignment.sent1 = re.sub('  ', ' ', alignment.sent1).rstrip(' ').lstrip(' ')
        complex = alignment.sent0.split(' ')
        complex_tags = [x[1] for x in nltk.pos_tag(complex)]
        complex = [(complex[i].lower(), i, complex_tags[i]) for i in range(len(complex))]
        simple = alignment.sent1.split(' ')
        simple_tags = [x[1] for x in nltk.pos_tag(simple)]
        simple = [(simple[i].lower(), i, simple_tags[i], simple[i][0]) for i in range(len(simple))]
        num_art_0 = sum(x[0] in ARTICLES for x in simple)
        num_art_1 = sum(x[0] in ARTICLES for x in complex)
        if num_art_0 != num_art_1:
            continue
        complex_only = [x for x in complex if x[0] not in [y[0] for y in simple] and x[0] and re.match(r'.*[a-z0-9].*', x[0]) and x[0] not in ARTICLES and x[2] != "NNP"]
        simple_only = [x for x in simple if x[0] not in [y[0] for y in complex] and x[0] and re.match(r'.*[a-z0-9].*', x[0]) and x[0] not in ARTICLES and x[2] != "NNP"]
        if len(complex_only) != 1 or len(simple_only) != 1:
            if len(complex_only) == len(simple_only) and len(complex_only) != 0:
                continue
                # sultan_aligner(alignment.sent0.split(' '), alignment.sent1.split(' '), complex_tags, simple_tags, len(simple_only))
            continue
        if complex_only[0][1] != simple_only[0][1]:
            # print(complex_only, simple_only, alignment.sent0)
            continue
        if complex_only[0][0] in STOPWORDS or simple_only[0][0] in STOPWORDS:
            continue
        if simple_only[0][2][0] != complex_only[0][2][0]:
            # print("parts of speech are different: " + str(simple_only) + "\t" + str(complex_only))
            pass
        if smart_lemmatize(complex_only[0][0], complex_only[0][2]) == smart_lemmatize(simple_only[0][0], simple_only[0][2]):
            continue
        # if not re.match('[ -~]*[a-z][ -~]*', complex_only[0][0]) or not re.match('[ -~]*[a-z][ -~]*', simple_only[0][0]):
            # continue
        key = re.sub('[^a-z]', '', alignment.sent0.rstrip('\n').casefold())
        if key in all_alignments:
            already_found = False
            for x in all_alignments[key]:
                if x[0] == alignment.sent0.split(' ')[complex_only[0][1]] and x[1] == alignment.sent1.split(' ')[simple_only[0][1]]:
                    already_found = True
                    break
            if not already_found:
                # print("Difference: " + " ".join([all_alignments[key][0][0], alignment.sent0.split(' ')[complex_only[0][1]], all_alignments[key][0][1], alignment.sent1.split(' ')[simple_only[0][1]]]))
                all_alignments[key].append([alignment.sent0.split(' ')[complex_only[0][1]], alignment.sent1.split(' ')[simple_only[0][1]]])
            else:
                continue
            continue
        else:
            all_alignments[key] = [[alignment.sent0.split(' ')[complex_only[0][1]], alignment.sent1.split(' ')[simple_only[0][1]]]]

        key = re.sub('[^a-z]', '', alignment.sent1.rstrip('\n').casefold())
        if key in all_alignments:
            already_found = False
            for x in all_alignments[key]:
                if x[0] == alignment.sent0.split(' ')[complex_only[0][1]] and x[
                    1] == alignment.sent1.split(' ')[simple_only[0][1]]:
                    already_found = True
                    break
            if not already_found:
                # print("Difference: " + " ".join([all_alignments[key][0][0], alignment.sent0.split(' ')[complex_only[0][1]], all_alignments[key][0][1], alignment.sent1.split(' ')[simple_only[0][1]]]))
                all_alignments[key].append(
                    [alignment.sent0.split(' ')[complex_only[0][1]],
                     alignment.sent1.split(' ')[simple_only[0][1]]])
            else:
                continue
            continue
        else:
            all_alignments[key] = [
                [alignment.sent0.split(' ')[complex_only[0][1]],
                 alignment.sent1.split(' ')[simple_only[0][1]]]]

        if debug:
            print(slug + "\t" + str(alignment.ind0) + "\t" + str(
                alignment.ind1))
            print(alignment.sent0.split(' ')[complex_only[0][1]] + '\t' + str(complex_only[0][1]) + '\t' + alignment.sent0.rstrip('\n'))
            print(alignment.sent1.split(' ')[simple_only[0][1]] + '\t' + str(simple_only[0][1]) + '\t' + alignment.sent1.rstrip('\n') + "\n")
            pass
        # file.write(slug + "\t" + str(alignment.ind0) + "\t" + str(
                # alignment.ind1) + "\n")
        # file.write(alignment.sent0.split(' ')[complex_only[0][1]] + '\t' + str(complex_only[0][1]) + '\t' + alignment.sent0.rstrip('\n') + "\n")
        # file.write(alignment.sent1.split(' ')[simple_only[0][1]] + '\t' + str(simple_only[0][1]) + '\t' + alignment.sent1.rstrip('\n') + "\n\n")
        als.append((alignment.ind0, alignment.ind1))
        file.write(alignment.sent0.split(' ')[complex_only[0][1]] + '\t' + str(
            complex_only[0][1]) + '\t1\t' + alignment.sent0.rstrip('\n') + "\t" + slug + "\t" + str(alignment.ind0) + "\t" + alignment.sent1.split(' ')[simple_only[0][1]] + '\n')
        complex = [x for x in complex if x[1] != complex_only[0][1]]
        complex = [x for x in complex if x[0].lower() not in STOPWORDS and re.match('.*[a-zA-Z].*', x[0]) and x[2] != "NNP"]
        for other in complex:
            file.write(alignment.sent0.split(' ')[other[1]] + '\t' + str(other[1]) + '\t0\t' + alignment.sent0.rstrip('\n') + "\t" + slug + "\t" + str(alignment.ind0) + '\t' + alignment.sent0.split(' ')[other[1]] + "\n")
    count = count_max_yield(als)
    if len(als) != count:
        print("Out of " + str(len(als)) + " alignments only " + str(count) + " would have been possible")
    return count, len(als)


def count_max_yield(als):
    max = 0
    if len(als) < 2:
        return len(als)
    for i in range(len(als)):
        passing = []
        for j in range(i + 1, len(als)):
            if (als[i][0] < als[j][0]) == (als[i][1] < als[j][1]):
                passing.append(als[j])
        curr = count_max_yield(passing)
        if curr > max:
            max = curr
    return 1 + max


if __name__ == "__main__":
    count = 0
    als = 0
    all_alignments = {}
    with open("/home/nlp/newsela/ALIGNMENTS.txt", "w") as file:
        i = 0
        info = loadMetafile()
        nSlugs = 0
        nToAlign = -1
        levels = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)]
        while (i < len(info)) and ((nToAlign == -1) or (nSlugs < nToAlign)):
            artLow = i  # first article with this slug
            slug = info[i]['slug']
            nSlugs += 1
            # if nToAlign == -1:
                # print(
                    # "Processing slug... " + slug + ' ' + str(round(i / float(len(info)) * 100, 3)) + '% of the task completed')
            while i < len(info) and slug == info[i]['slug']:
                i += 1
            artHi = i  # one more than the number of the highest article with this slug
            for level in levels:
                if level[1] < artHi - artLow:
                    sentpairs = get_aligned_sentences(info, slug, level[0], level[1])
                    # output_alignments(file, sentpairs, slug + "\t" + "\t".join([str(x) for x in level]))
                    c, a = output_alignments(file, sentpairs, slug + "\t" + "\t".join([str(x) for x in level]), all_alignments, abs(level[0] - level[1]) != 1)
                    count += c
                    als += a
                else:
                    print("No such level: " + str(level[1]) + " in article: " + slug)
    print("Out of " + str(als) + " alignments only " + str(count) + " would have been possible")
