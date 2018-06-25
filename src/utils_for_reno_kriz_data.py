"""
Module for searching for sentences in the newsela corpus
"""

from newselautil import *
from classpaths import *
import spacy
import prepDataOneHot as prepData
import bz2
import pickle
import alignutils

EXT = ".spacy"  # spaCy extension
sentences = {}  # all of the sentences loaded as a dictionary
# nlp = spacy.load('en_core_web_sm')
# spaCy model that the authors of the article most likely used


def create_spacy_corpus():
    """
    Tokenized all the 0-level articles using spacy and saves them in the
    .spacy format. Note: in order to keep track of sentences' indexes the
    .tok files are tokenized. This means that the output of the tokenization
    has to undergo some changes so that it looks as if the .txt files were
    directly tokenized
    :return:
    """
    info = loadMetafile()
    i = 0
    while i < len(info):
        slug = info[i]['slug']
        with io.open(path.BASEDIR + '/articles/' + info[i]['filename'] + '.tok',
                     mode='r', encoding='utf-8') as fd:
            lines = fd.readlines()
            create_spacy_file(info[i]['filename'], lines)
            print("Slug:  " + slug + ' ' + str(round(
                i / float(len(info)) * 100, 3)) + '% of the task completed')
        while i < len(info) and slug == info[i]['slug']:
            i += 1


def create_spacy_file(filename, lines):
    """
    Tokenized a single article and saves it in a .spacy format
    :param filename: The name of the file tokenized
    :param lines:    The contents of the original file
    :return:
    """
    PARPREFIX = "@PGPH "  # Delimits paragraphs in FILE.tok
    with io.open(path.BASEDIR + '/articles/' + filename + EXT, 'w') as file:
        for line in lines:
            if line[0:len(PARPREFIX)] == PARPREFIX or line[0] == "#":
                continue
            line = ' '.join([token.text for token in nlp(line)])
            line = re.sub('&', '& amp ;', line)
            line = re.sub("`", "'", line)
            line = re.sub(" ' re ", " 're ", line)
            line = re.sub(" ' s ", " 's ", line)
            line = re.sub('<', '& lt;', line)
            line = re.sub('>', '& gt ;', line)
            file.write(line)


def load_spacy_corpus():
    """
    Loads all of the corpus (0-level articles) in the sentences dictionary
    :return:
    """
    info = loadMetafile()
    i = 0
    while i < len(info):
        slug = info[i]['slug']
        with io.open(path.BASEDIR + '/articles/' + info[i]['filename'] + EXT,
                     mode='r', encoding='utf-8') as fd:
            lines = fd.readlines()
            for j in range(len(lines)):
                line = re.sub(r'[^a-zA-Z]', '', lines[j]).lower()
                # This way it is much more probable that the sentences from
                # .tok files will match those from the sentences dictionary
                if line not in sentences:
                    sentences[line] = [(info[i]['slug'], j)]
                else:
                    sentences[line].append((info[i]['slug'], j))
        while i < len(info) and slug == info[i]['slug']:
            i += 1


def detect_sentences():
    """
    Determine to which article a sentence from KRIZ_PAPER_FILE belongs to and
    write this new information to a new file
    :return:
    """
    load_spacy_corpus()
    with io.open(KRIZ_PAPER_FILE, 'r') as file:
        lines = file.readlines()
    with io.open(KRIZ_PAPER_FILE.split('.')[0] + '_supplied.txt', 'w') as file:
        i = 0
        lines_total = 0
        lines_determined = 0
        previous = ""
        while i < len(lines):
            line = lines[i].split('\t')[-1]
            line = re.sub(r'[^a-zA-Z]', '', line)
            line = line.lower()
            lines_total += 1
            if line not in sentences:
                if i == 0 or \
                        lines[i].split('\t')[-1] != lines[i-1].split('\t')[-1]:
                    # print("Cannot find: " + lines[i].split('\t')[-1])
                    pass
                i += 1
                continue

            if len(sentences[line]) > 1:
                chosen = None
                for version in sentences[line]:
                    if previous == version[0]:
                        sentences[line] = [version]
                        # print("Chosen: " + str(version[0]) + '\n')
                        chosen = True
                        break
                if not chosen:
                    # print("Line appears in many files: " +
                    #     lines[i].split('\t')[-1][:-1])
                    # print("These files are: " + ' '.join(
                    #     x[0] for x in sentences[line]))
                    i += 1
                    continue

            file.writelines(lines[i].rstrip('\n') + '\t' +
                            sentences[line][0][0] + '\t' +
                            str(sentences[line][0][1]) + '\n')
            previous = sentences[line][0][0]
            lines_determined += 1
            i += 1
        print('Found files for ' + str(lines_determined) + ' out of ' +
              str(lines_total) + ' lines')


def try_word(word, marked, ind):
    """
    See if the marked[ind] equal the word.
    :param word:    A word from the Reno Kriz data file
    :param marked:  The alignment as a list of tokens
    :param ind:     The id of the token from marked to match with word
    :return: A tuple of two booleans. The first is True, if the two words match.
    The second is true, if the marked[ind] is marked as complex in the alignment
    """
    complex = False
    if ind < -len(marked) or ind >= len(marked):
        return False, False
    if marked[ind][0] == '_':
        marked[ind] = marked[ind][1:-1]
        complex = True
    if re.match('.*[^ -~].*', word):
        print("Weird word: " + word)
    elif marked[ind] == word:
        return True, complex
    return False, complex


def match_word(word, ind, alignment, original):
    """
    Try to match teh alignment outpu tand the output of the Kriz paper.
    :param word:
    :param ind:     index of the word according to Reno Kriz paper data
    :param score:
    :param alignmentnt: list of tokens or an alignemnt object
    :param original:
    :return: A tuple of two booleans and an integer. The first boolean is True
    if the corresponding word was found in the alignment output. The second is
    True if the word is considered complex by the alignments program. The
    integer (can be negative) is the index of the word inside alignment
    """
    word = word.lower()
    if isinstance(alignment, alignutils.Alignment):
        marked = alignment.mark_simplified(additionalTokenizer=False)
    else:
        marked = alignment
    match, complex = try_word(word, marked, ind)
    if match:
        return True, complex, ind
    match, complex = try_word(word, marked, ind - len(original.split(' ')))
    if match:
        return True, complex, ind - len(original.split(' '))

    new_ind = -1
    i = 0
    while i < len(marked):
        match, complex_candidate = try_word(word, marked, i)
        if match:
            if new_ind != -1:
                print("Two or more instances ")
                return False, None, None
            complex = complex_candidate
            new_ind = i
        i += 1
    if new_ind == -1:
        print("No instances")
        return False, None, None
    return True, complex, new_ind


def compare_alignments(level_to_compare):
    """
    Print various statistics about how the data from the KRIZ PAPER differs
    from that obtained via the alignment algorithm
    :param level_to_compare: Lower level of alignment to use
    :return:
    """
    info = loadMetafile()
    aligned_files = {}
    # a dictionary, where each slug corresponds to a list of alignments
    sentences_total = 0
    sentences_aligned = 0
    words_in_aligned_sentences = 0
    words_aligned = 0
    levels = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
              [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    with io.open(KRIZ_PAPER_FILE.split('.')[0] + '_supplied.txt') as file:
        lines = file.readlines()
    prev = ("", 0)
    for line in lines:
        word, w_ind, score, sent, slug, s_ind = line.rstrip('\n').split('\t')
        w_ind = int(w_ind)
        s_ind = int(s_ind)
        score = int(score)

        if slug != prev[0] or s_ind != prev[1]:
            sentences_total += 1

        if slug not in aligned_files:
            aligned_files[slug] = alignutils.get_aligned_sentences(
                info, slug, 0, level_to_compare, use_spacy=True)
        slug_aligned = aligned_files[slug]
        for sent_al in slug_aligned:
            if sent_al.ind0 == s_ind:
                if slug != prev[0] or s_ind != prev[1]:
                    sentences_aligned += 1
                words_in_aligned_sentences += 1
                matching = match_word(word, w_ind, sent_al, sent)
                if matching[0]:
                    words_aligned += 1
                    if matching[1]:
                        levels[score][0] += 1
                    else:
                        levels[score][1] += 1
                break
        prev = (slug, s_ind)
    print('Out of ' + str(sentences_total) + ', ' + str(sentences_aligned) +
          ' sentences are aligned')
    print('Found words for ' + str(words_aligned) + ' out of the total of ' +
          str(words_in_aligned_sentences))
    for i in range(len(levels)):
        if levels[i][0] + levels[i][1] != 0:
            print("Level " + str(i) + ": " + str(round(float(levels[i][0])/(levels[i][0] + levels[i][1]), 2)))
        else:
            break
    return aligned_files


def create_idx_file():
    """
    Create an idx file for the data from the Reno Kriz paper
    :return:
    """
    with io.open(KRIZ_PAPER_FILE.split('.')[0] + '_supplied.txt') as file:
        lines = file.readlines()
    previous = ""
    output = []
    for line in lines:
        filename = line.split('\t')[-2] + '.en.0'
        if filename == previous:
            continue
        previous = filename
        pars = getTokParagraphs({"filename": filename + ".txt"},
                                separateBySemicolon=False)
        info = [filename, str(len(pars))] + [str(len(x)) for x in pars]
        output.append(' '.join(info))
    with io.open(KRIZ_IDX_FILE, 'w') as file:
        file.writelines(str(len(output)) + '\n' + '\n'.join(output))


def create_bz2_file():
    """
    Create a bz2 file for teh data from teh Reno Kriz paper
    :return:
    """
    with bz2.BZ2File(nnetFile, 'r') as handle:
        (invoc, _) = pickle.load(handle)
    filenames, npars, sindlst = prepData.read_index_file(KRIZ_IDX_FILE)
    word_to_index = dict([(w, i) for i, w in enumerate(invoc)])
    pars = []
    for file in filenames:
        ps = getTokParagraphs({"filename": file + ".txt"},
                                separateBySemicolon=False)
        """with open(path.BASEDIR + '/articles/' + file + '.txt.spacy') as spacy_file:
            spacy_lines = spacy_file.readlines()
            abs_id = -1
            for p in ps:
                for i in range(len(p)):
                    abs_id += 1
                    p[i] = spacy_lines[abs_id].rstrip('\n ')"""

        pars += ps
    # all_sents, _ = prepData.create_sentences(pars, word_to_index, tokenize=False)
    all_sents, _ = prepData.create_sentences(pars, word_to_index,
                                             tokenize=True)
    with bz2.BZ2File(KRIZ_BZ2_FILE, 'w') as fd:
        pickle.dump((invoc, all_sents), fd)


def get_complexity_data():
    """
    This method is created to be used inside simpProbabilities.py. The method's
    output is the quadruple list (first level is a dictionary).
    result[filename_from_.idx][sent_id][1][word_id] is the word
    result[filename_from_.idx][sent_id][1][word_id] is the complexity score
    assigned to it
    :return:
    """
    filenames, _, _ = prepData.read_index_file(KRIZ_IDX_FILE)
    pars = {}

    # loading spaCy sentences to pars
    for file in filenames:
        tokenized = getTokParagraphs({"filename": file + ".txt"},
                                separateBySemicolon=False)
        with open(path.BASEDIR + '/articles/' + file + '.txt.spacy') as spacy_file:
            spacy_lines = spacy_file.readlines()
        abs_id = -1
        ps = []
        for p in tokenized:
            for i in range(len(p)):
                abs_id += 1
                # ps.append([[x.lower() for x in spacy_lines[abs_id].rstrip(' \n').split(' ')],
                           # [-1 for x in spacy_lines[abs_id].split(' ')]])
                ps.append([[x.lower() for x in tokenize(p[i])],
                           [-1 for x in tokenize(p[i])]])
        pars[file] = ps

    # marking the words for which the complexity is known
    with io.open(KRIZ_PAPER_FILE.split('.')[0] + '_supplied.txt') as file:
        lines = file.readlines()
    for line in lines:
        word, w_ind, score, sent, slug, s_ind = line.rstrip('\n').split('\t')
        w_ind = int(w_ind)
        s_ind = int(s_ind)
        score = int(score)
        nn_sentence = pars[slug + '.en.0'][s_ind]
        match, _, ind = match_word(word, w_ind, nn_sentence[0], sent)
        if match:
            nn_sentence[1][ind] = score

    return pars


if __name__ == "__main__":
    # create_spacy_corpus()
    detect_sentences()
    compare_alignments(3)
    # create_idx_file()
    # create_bz2_file()
