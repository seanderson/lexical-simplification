from lexenstein.morphadorner import MorphAdornerToolkit  # for syllable count
import numpy
import statistics
import classpaths as paths
from nltk.corpus import wordnet as wn
from newsela_pos import *  # POS tagger
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim


Lemmatizer = WordNetLemmatizer()
LEXICON = "/home/nlp/Lexicons/ALL.tsv"
N_GRAM_DIRECTORY = "/home/nlp/corpora/n-grams/"
ORIGINAL_DATA = "/home/nlp/corpora/cwi/traindevset/english/WikiNews_Train.tsv"
# the data in "Chris" format, i.e. a line with tab-separated values:
# word  ind score   sentence    substituition (the latter is optional)
FEATURE_DIR = "/home/nlp/corpora/cwi/traindevset/english/Full_Features_Sasha/WikiNews_Train/"
# the directory to which all the numpy arrays will be stored
EMB_MODEL = paths.NEWSELA_ALIGNED + "model.bin"
# current model is trained with vectors of size 500, window = 5
# alpha = 0.01, min_alpha = 1.0e-9, negative sampling = 5. The third epoch is
# taken because it shows the best score on SimLex-999
EMB_SIZE = 500
N_ALT = 0
# the number of substitution candidates to add (the substitutions are chosen
# from the nearest word vecors that bear the same POS tag). This constant should
# be zero, if the document in question contains phrases
TOP_N = 20
# the number of most-similar words to grab when looking for alternatives
# Out of these only those words will be added to alternatives that have the same
# POS tag as the target word
BINARY = lambda x: 0 if int(x) <= 3  else 1

"""
ORIGINAL_DATA = \
    paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt"
FEATURE_DIR = paths.NEWSELA_COMPLEX + "features/"
EMB_MODEL = paths.NEWSELA_COMPLEX + "model.bin"
EMB_SIZE = 500
BINARY = lambda x: 0 if int(x) <= 3  else 1 
"""


class CustomFeatureEstimator:

    TAG_TO_NUM = {'A': 5, 'MD': 2, 'PDT': 9, 'RP': 8, 'IN': 6, '-RRB-': 7,
                  'CC': 11, 'LS': 17, 'J': 3, 'SYM': 18, 'N': 1, 'P': 12,
                  'UH': 16, 'W': 15, 'V': 19, '-LRB-': 13, 'DT': 10, 'CD': 4,
                  'FW': 14, 'PHRASE': 0}
    NUM_TO_TAG = {0: 'PHRASE', 1: 'N', 2: 'MD', 3: 'J', 4: 'CD', 5: 'A',
                  6: 'IN', 7: '-RRB-', 8: 'RP', 9: 'PDT', 10: 'DT', 11: 'CC',
                  12: 'P', 13: '-LRB-', 14: 'FW', 15: 'W', 16: 'UH', 17: 'LS',
                  18: 'SYM', 19: 'V'}

    VOWELS = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y']
    VOWEL_REGEX = re.compile(r'[AEIOUYaeiouy]')
    NON_ASCII = '[^\x00-\x7F]'

    def __init__(self, feature_names=[]):
        """
        Creates an instance of the FeatureEstimator class.
        :param feature_names: the features to calculate when running
        calculate_features
        """
        self.all_features = {}
        self.fill_all_features()
        self.results = {}
        self.features = [self.all_features[name] for name in feature_names]
        # self.features is an array of features that will be used during this
        # particular run

        if N_ALT > 0:
            if feature_names[0] != 'wv' and (
                    feature_names[1] != 'wv' or feature_names[0] != 'POS'):
                print('If alternatives are to be calculated, place POS and wv '
                      'in the beginning')
                # exit(0)

        for i in range(len(self.features)):
            for dependency in self.features[i]['dep']:
                # some features require that another feature is executed
                # beforehand (e.g. word vectors need to know the POS tags)
                if dependency not in [x['name'] for x in self.features[:i]]:
                    print("Feature " + dependency +
                          " should come before feature " +
                          self.features[i]['name'])
                    try:
                        self.results[dependency] = \
                            numpy.load(FEATURE_DIR + dependency + '.npy')
                    except:
                        exit(-1)
                    print("A saved version of the former is loaded from a file")

    def fill_all_features(self):
        """
        Create the "all_features" dictionary that can be used to look up feature
        names and associated functions
        :return:
        """
        self.all_features = {
            "POS": {"func": self.pos_tag_feature, "dep": []},
            "sent_syllab": {"func": self.sent_syllable_feature, "dep": []},
            "word_syllab": {"func": self.word_syllable_feature, "dep": []},
            "word_count": {"func": self.word_count_feature, "dep": []},
            "mean_word_length": {"func": self.mean_word_length_feature,
                                 "dep": []},
            "synset_count": {"func": self.synset_count_feature, "dep": []},
            "synonym_count": {"func": self.synonym_count_feature, "dep": []},
            "labels": {"func": self.get_labels, "dep": []},
            "wv": {"func": self.word_embeddings_feature, "dep": ["POS"]},
            "hit": {"func": self.hit_freqency_feature, "dep": []},
            "1-gram": {"func": lambda x: self.n_gram_feature(x, 1), "dep": []},
            "2-gram": {"func": lambda x: self.n_gram_feature(x, 2), "dep": []},
            "3-gram": {"func": lambda x: self.n_gram_feature(x, 3), "dep": []},
            "4-gram": {"func": lambda x: self.n_gram_feature(x, 4), "dep": []},
            "5-gram": {"func": lambda x: self.n_gram_feature(x, 5), "dep": []},
            "vowel_count": {"func": self.vowel_count_feature, "dep": []},
            "lexicon": {"func": self.lexicon_feature, "dep": ["POS"]}
        }
        for key in self.all_features:
            self.all_features[key]["name"] = key

    def calculate_features(self, data):
        """
        Calculate all the features in self.features and write the results as
        numpy arrays
        :param data: a list of dictionary objects with fields "sent", "words",
        and "inds".
            "sent" must be a sentence
            "words" - a list of target words (not necessary from a sentence)
            "indes" - their indexes within the sentence (if such exist)
        :return:
        """
        for feature in self.features:
            print("Assessing " + feature["name"])
            self.results[feature["name"]] = feature["func"](data)
            numpy.save(FEATURE_DIR + feature["name"],
                       numpy.array(self.results[feature["name"]]))
        print('Done')

    def load_features(self):
        """
        Simply loads all the specified features from corresponding files
        :return: a numpy matrix
        """
        result = []
        for feature in self.features:
            if feature["name"] == "labels":
                print("Labels will not be appended to the feature list."
                      "\nUse load_labels to get the labels")
                continue
            result.append(numpy.load(FEATURE_DIR + feature["name"] + ".npy"))
        result = numpy.concatenate(result, axis=1)
        return result

    def load_labels(self):
        """
        Load teh labels as a numpy array
        :return:
        """
        return numpy.load(FEATURE_DIR + "labels.npy")

    def sent_syllable_feature(self, data):
        """
        Calculate the average number of syllables per word in each sentence
        :param data: See the entry for calculate_features
        :return:
        """
        input = []
        n_of_words = []
        for line in data:
            tmp = [x for x in line['sent'].split(' ') if
                   re.match('.*[a-zA-Z].*', x)]
            n_of_words.append(len(tmp))
            input += tmp
        output = self.syllabify(input)
        ind = 0
        result = []
        for i in range(len(data)):
            value = float(sum(output[ind: ind+n_of_words[i]])) / n_of_words[i]
            result.append(numpy.array([value]))
            ind += n_of_words[i]
        return result

    def word_syllable_feature(self, data):
        """
        Calculate the number of syllables in each of teh target words
        :param data: See the entry for calculate_features
        :return:
        """
        input = []
        for line in data:
            input += line['words']
        n_candidates = [len(line['words']) for line in data]
        output = self.syllabify(input)
        ind = 0
        result = []
        for i in range(len(data)):
            n_of_words = n_candidates[i]  # N of words in sent
            result.append(numpy.zeros(N_ALT + 1))
            if n_of_words != 0:
                for j in range(len(result[-1])):
                    result[-1][j] = output[ind + j]
            ind += n_of_words
        return result

    def syllabify(self, list_of_words):
        """
        Syllabifies the input list of strings and return a
        list of syllable-counts
        :param list_of_words:
        :return:
        """
        mat = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)
        out = mat.splitSyllables(list_of_words)
        out = [o.decode("latin1").replace(' ', '-') for o in out]
        return [len(o.split('-')) for o in out if len(o.strip()) > 0]

    def word_count_feature(self, data):
        """
        Calculate the number of words in each sentence
        :param data: See the entry for calculate_features
        :return:
        """
        return [numpy.array([len(line['sent'].split(' '))]) for line in data]

    def mean_word_length_feature(self, data):
        """
        Calculate the mean word length per sentence
        :param data: See the entry for calculate_features
        :return:
        """
        return [numpy.array([statistics.mean([len(x) for x in line['sent'].split(' ') if
                          re.match('.*[a-zA-Z].*', x)])]) for line in data]

    def synset_count_feature(self, data):
        """
        Calculate the numer of wordnet synsets for each target word
        :param data: See the entry for calculate_features
        :return:
        """
        dict = {}
        result = []
        for line in data:
            result.append(numpy.zeros(N_ALT + 1))
            for j in range(len(line['words'])):
                word = line['words'][j].lower()
                if word not in dict:
                    try:
                        dict[word] = len(wn.synsets(word))
                    except UnicodeDecodeError:
                        dict[word] = 0
                result[-1][j] = dict[word]
        return result

    def synonym_count_feature(self, data):
        """
        Calculate the numer of wordnet synonyms for each target word
        :param data: See the entry for calculate_features
        :return:
        """
        dict = {}
        result = []
        for line in data:
            result.append(numpy.zeros(N_ALT + 1))
            for j in range(len(line['words'])):
                word = line['words'][j].lower()
                if word not in dict:
                    try:
                        senses = wn.synsets(word)
                    except UnicodeDecodeError:
                        senses = []
                    dict[word] = 0
                    for sense in senses:
                        dict[word] += len(sense.lemmas())
                        # TODO: is it the way to derive the number of synonyms?
                result[-1][j] = dict[word]
        return result

    def pos_tag_feature(self, data):
        """
        Find out the POS tag of each target word that is in the sentence
        :param data:  See the entry for calculate_features
        :return:
        """
        tags = get_tags(data)
        result = []
        next_id = len(self.TAG_TO_NUM) + 1
        for tag in tags:
            if tag not in self.TAG_TO_NUM:
                self.TAG_TO_NUM[tag] = next_id
                self.NUM_TO_TAG[next_id] = tag
                next_id += 1
            result.append(numpy.array([self.TAG_TO_NUM[tag]]))
        return result

    def word_embeddings_feature(self, data):
        """
        Get word embeddings from the default model
        :param data:  See the entry for calculate_features
        :return:
        """
        result = []
        model = gensim.models.KeyedVectors.load_word2vec_format(EMB_MODEL,
                                                                binary=True)
        file = open(FEATURE_DIR + 'substitutions', 'w')
        for i in range(len(data)):
            if data[i]['phrase']:
                result.append(numpy.zeros(EMB_SIZE))
                continue
            tag = self.NUM_TO_TAG[self.results['POS'][i][0]]
            target = data[i]['words'][0].lower() + '_' + tag
            result.append([])
            if target not in model.vocab:
                result[-1].append(numpy.zeros(EMB_SIZE * (N_ALT + 1) + N_ALT))
                file.write('\n')
            else:
                if i % 100 == 0:  # Debuggin progress
                    print(i)
                result[-1].append(model[target])
                if N_ALT > 0:  # in this case possible substitutions will be
                    # selected based on the topn most similar word vectors
                    closest = model.wv.most_similar(positive=[target], topn=TOP_N)
                    j = 0
                    ind = 0
                    cosines = numpy.zeros(N_ALT)
                    while j < len(closest) and ind < N_ALT:
                        substitution, cosine = closest[j]
                        substitution = substitution.split('_')
                        word = '_'.join(substitution[:-1])
                        tag_check = substitution[-1]
                        if tag_check == tag:
                            cosines[ind] = cosine
                            file.write(re.sub(self.NON_ASCII, '', word) + '\t')
                            data[i]['words'].append(re.sub(self.NON_ASCII, '', word))
                            data[i]['inds'].append(data[i]['inds'][0])
                            if word + '_' + tag in model.vocab:
                                result[-1].append(model[word + '_' + tag])
                            else:
                                result[-1].append(numpy.zeros(EMB_SIZE))
                            ind += 1
                        j += 1
                    file.write('\n')
                    while len(result[-1]) != N_ALT + 1:
                        result[-1].append(numpy.zeros(EMB_SIZE))
                    result[-1].append(cosines)
            result[-1] = numpy.concatenate(result[-1])
        return result

    def get_labels(self, data):
        """
        Create a list of labels (scores)
        :param data:
        :return:
        """
        if N_ALT == -1:
            result = []
            for line in data:
                if line['subst'] in line['words']:
                    result.append(line['words'].index(line['subst']))
                else:
                    result.append(N_ALT + 1)
            return result
        return [line['score'] for line in data]

    def n_gram_feature(self, data, n):
        """
        Get google n-gram-score
        :param data:
        :param n:
        :return:
        """
        result = numpy.zeros((len(data), (N_ALT + 1) * n))
        dictionary = {}
        # dictionary of values that are to be looked up in the n-grams
        for i in range(len(data)):
            line = data[i]
            for j in range(len(line['inds'])):
                for k in range(n):
                    id = line['inds'][j]
                    if id - k < 0 and id - k + n -1 >= len(line['sent'].split(' ')):
                        continue
                    query = ' '.join(line['sent'].split(' ')[id - k: id])
                    if query != "":
                        query += " "
                    query += line['words'][j]
                    if id + 1 != id - k + n:
                        query += " "
                    query += ' '.join(line['sent'].split(' ')[id + 1: id - k + n])
                    query = re.sub('`', '\'', query)
                    if query not in dictionary:
                        dictionary[query] = []
                    dictionary[query].append((i, j * n + k))
        self.fill_dictionary_from_n_grams(dictionary, result, n)
        return result

    def fill_dictionary_from_n_grams(self, dictionary, result, n):
        """
        Look for those n-grams that are in the dictionary and fill with them
        the result
        :param n:
        :return:
        """
        N_OF_FILES = [0, 1, 32, 98, 132, 118]
        # number of files for a google n-gram vocabulary
        if n == 1:
            with open(N_GRAM_DIRECTORY + "1gms/vocab") as file:
                for line in file:
                    ngram, count = line.rstrip('\n').split('\t')
                    if ngram in dictionary:
                        for entry in dictionary[ngram]:
                            result[entry[0]][entry[1]] = int(count)
            return
        for i in range(N_OF_FILES[int(n)]):
            print (i)
            str_id = '0' * (4 - len(str(i))) + str(i)
            with open(N_GRAM_DIRECTORY + str(n) + "gms/" + str(n) + "gm-" + str_id) as file:
                for line in file:
                    ngram, count = line.rstrip('\n').split('\t')
                    if ngram in dictionary:
                        for entry in dictionary[ngram]:
                            result[entry[0]][entry[1]] = int(count)

    def hit_freqency_feature(self, data):
        """
        Return the frequency of a particular word/phrase within the hit
        :param data:
        :return:
        """
        hits = {}  # dictionary that stores list of line indexes corresponing to
        # a given hit_id
        result = []
        for i in range(len(data)):
            result.append(numpy.zeros((N_ALT + 1) * 2))
            # result[i][j * 2] is the count of token[j] in lines with
            # hit_id = data[i]['hit']
            # result[i][j * 2 + 1] is the number of such lines
            if data[i]['hit'] is None:
                continue
            tokens = [x.lower() for x in data[i]['words'] + data[i]['phrase']]
            # tokens for which the hit frequency is to be counted
            newsent = False
            if data[i]['hit'] not in hits:
                hits[data[i]['hit']] = [i]
            else:
                sents = [data[lid]['sent'] for lid in hits[data[i]['hit']]]
                if data[i]['sent'] not in sents:
                    newsent = True
                hits[data[i]['hit']].append(i)
            sents = []
            for line_id in hits[data[i]['hit']]:
                if data[line_id]['sent'] not in sents:
                    sents.append(data[line_id]['sent'])
                    for j in range(len(tokens)):
                        result[i][j * 2] += data[line_id]['sent'].lower().count(tokens[j])
                        result[i][j * 2 + 1] += len(data[line_id]['sent'].split(' '))
                if not newsent or i == line_id:
                    continue
                tmp = [x.lower() for x in data[line_id]['words'] + data[line_id]['phrase']]
                for j in range(len(tmp)):
                    result[line_id][j * 2] += data[i]['sent'].lower().count(tmp[j])
                    result[line_id][j * 2 + 1] += len(data[i]['sent'].split(' '))
        return result

    def vowel_count_feature(self, data):
        """
        Calculate the number of vowels in the target words/phrase
        :param data:
        :return:
        """
        result = []
        for i in range(len(data)):
            result.append(numpy.zeros(N_ALT + 1))
            tokens = data[i]['words'] + data[i]['phrase']
            for j in range(len(tokens)):
                result[-1][j] = len(self.VOWEL_REGEX.findall(tokens[j]))
        return result

    def lexicon_feature(self, data):
        """
        Read a lexicon and look if a specific word is inside of teh lexicon or
        not
        :param data:
        :return:
        """
        with open(LEXICON) as file:
            lex = [line.rstrip('\n').split('\t') for line in file.readlines()[1:]]
        size = len(lex[0]) - 1  # number of entries per word in the lexicon
        lex = {line[0]: [int(float(score)) for score in line[1:]] for line in lex}
        result = []
        for i in range(len(data)):
            line = data[i]
            result.append(numpy.zeros((N_ALT + 1) * 2 * size))
            for j in range(len(line['words'])):
                word = line['words'][j].lower()
                tag = self.NUM_TO_TAG[self.results['POS'][i][0]]

                if word + '_' + tag in lex:
                    for k in range(size):
                        result[-1][j * size * 2 + k] = lex[word + '_' + tag][k]
                elif smart_lemmatize(word, tag) + '_' + tag in lex:
                    for k in range(size):
                        result[-1][j * size * 2 + k] = lex[smart_lemmatize(word, tag) + '_' + tag][k]

                if word in lex:
                    for k in range(size):
                        result[-1][j * size * 2 + size + k] = lex[word][k]
                elif smart_lemmatize(word, tag) in lex:
                    for k in range(size):
                        result[-1][j * size * 2 + size + k] = lex[smart_lemmatize(word, tag)][k]
        return result


def get_raw_data():
    """
    Load the raw data (in Chris format) in memory
    :return:
    """
    with open(ORIGINAL_DATA) as file:
        lines = file.readlines()
    lines = [re.sub(CustomFeatureEstimator.NON_ASCII, '', line.rstrip('\n')).split('\t') for line in lines]
    lines = [{'words': [x[0]], 'sent': x[3], 'inds': [int(x[1])],
              'score':BINARY(x[2]), 'phrase': [], 'hit': None,
              'subst': x[-1]} for x in lines]
    for k in range(len(lines)):
        line = lines[k]
        while re.match(r'.*[ ]{2}.*', line['sent']):
            bad_id = line['sent'].split(' ').index('')
            for i in range(len(line['inds'])):
                if line['inds'][i] > bad_id:
                    line['inds'][i] -= 1
            tmp = line['sent'].split(' ')
            del tmp[bad_id]
            line['sent'] = ' '.join(tmp)
    LOADIT = False
    if N_ALT > 0 and LOADIT:
        with open(FEATURE_DIR + "substitutions") as file:
            subst = [re.sub(CustomFeatureEstimator.NON_ASCII, '', x.rstrip('\n')).split('\t')[:N_ALT] for x in file.readlines()]
        for i in range(len(subst)):
            for s in subst[i]:
                if s != "":
                    lines[i]['words'].append(s)
                    lines[i]['inds'].append(lines[i]['inds'][0])
    return lines


def get_cwi_data():
    """
    Load the data (in CWI Semeval format) into memory
    :return:
    """
    print(ORIGINAL_DATA)
    with open(ORIGINAL_DATA) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    result = []
    for i in range(len(lines)):
        if len(lines[i][4].split(' ')) > 1:
            # the target is a phrase
            continue
            ind1 = int(len(lines[i][1][:int(lines[i][2])].split(' ')))
            ind2 = int(len(lines[i][1][:int(lines[i][3])].split(' ')))
            result.append({'phrase': [lines[i][4]], 'words': [], 'hit': lines[i][0],
                        'sent': lines[i][1],
                        'score': int(lines[i][9]),
                        'inds': [], 'phrase_ind': [ind1, ind2]})
        else:
            # the target is a word
            sent, ind = tokenize(lines[i][1], int(lines[i][2]))
            ind = int(len(sent[:ind].split(' '))) - 1
            result.append({'words': [lines[i][4]], 'phrase': [],
                        'hit': lines[i][0],
                        'sent': sent,
                        'score': int(lines[i][9]), 'inds': [ind],
                        'phrase_ind': []})
    return result


def smart_lemmatize(word, treebank_tag):
    # print(word, treebank_tag)
    word = re.sub('[^a-zA-Z-]', '', word)
    if word == "":
        return word
    if treebank_tag.startswith('J'):
        return Lemmatizer.lemmatize(word, wordnet.ADJ)
    elif treebank_tag.startswith('V'):
        return Lemmatizer.lemmatize(word, wordnet.VERB)
    elif treebank_tag.startswith('N'):
        return Lemmatizer.lemmatize(word, wordnet.NOUN)
    elif treebank_tag.startswith('A'):
        return Lemmatizer.lemmatize(word, wordnet.ADV)
    else:
        return word


if __name__ == "__main__":
    fe = CustomFeatureEstimator(["POS", "hit", "sent_syllab", "word_syllab",
                                 "word_count", "mean_word_length", "synset_count",
                                 "synonym_count", "vowel_count", "1-gram",
                                 "2-gram", "3-gram", "4-gram", "5-gram", "lexicon",
                                 "wv", "labels"])
    # fe = CustomFeatureEstimator(["lexicon"])
    # fe.calculate_features(get_cwi_data())
    features = fe.load_features()
    labels = fe.load_labels()
    data = get_cwi_data()
    if len(data) != len(features):
        exit(-1)
    for i in range(100):
        line = data[i]
        print('\n')
        print(data[i]['sent'] + '\t' + str(data[i]['phrase']))
        print(str(data[i]['words']))
        # print(data[i]['subst'])
        print("POS: " + CustomFeatureEstimator.NUM_TO_TAG[features[i][0]])
        print("hit: " + str(features[i][1:1 + 2 * (N_ALT + 1)]))
        print("sent_syllab: " + str(features[i][1 + 2 * (N_ALT + 1)]))
        print("word_syllab: " + str(features[i][2 + 2 * (N_ALT + 1):2 + 3 * (N_ALT + 1)]))
        print("word_count: " + str(features[i][2 + 3 * (N_ALT + 1)]))
        print("mean_word_length: " + str(features[i][3 + 3 * (N_ALT + 1)]))
        print("synset_count: " + str(features[i][4 + 3 * (N_ALT + 1):4 + 4 * (N_ALT + 1)]))
        print("synonym_count: " + str(features[i][4 + 4 * (N_ALT + 1):4 + 5 * (N_ALT + 1)]))
        print("vowel_count: " + str(features[i][4 + 5 * (N_ALT + 1):4 + 6 * (N_ALT + 1)]))
        print("1-gram: " + str(features[i][4 + 6 * (N_ALT + 1):4 + 7 * (N_ALT + 1)]))
        print("2-gram: " + str(features[i][4 + 7 * (N_ALT + 1):4 + 9 * (N_ALT + 1)]))
        print("3-gram: " + str(features[i][4 + 9 * (N_ALT + 1):4 + 12 * (N_ALT + 1)]))
        print("4-gram: " + str(features[i][4 + 12 * (N_ALT + 1):4 + 16 * (N_ALT + 1)]))
        print("5-gram: " + str(features[i][4 + 16 * (N_ALT + 1):4 + 21 * (N_ALT + 1)]))
        if N_ALT > 0:
            print("cosines: " + str(features[i][-N_ALT:]))
        # print("wv: " + str(features[i][4 + 21 * (N_ALT + 1):4 + 521 * (N_ALT + 1)]))
        print("label: " + str(labels[i]))
