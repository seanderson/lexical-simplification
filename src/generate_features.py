from lexenstein.morphadorner import MorphAdornerToolkit  # for syllable count
import numpy
import statistics
import classpaths as paths
from nltk.corpus import wordnet as wn
from newsela_pos import *  # POS tagger
import gensim


CWI_DATA = "/home/nlp/corpora/cwi/traindevset/english/News_Dev.tsv"
N_GRAM_DIRECTORY = "/home/nlp/corpora/n-grams/"
ORIGINAL_DATA = paths.NEWSELA_ALIGNED + "dataset.txt"
# the data in "Chris" format, i.e. a line with tab-separated values:
# word  ind score   sentence    substituition (the latter is optional)
FEATURE_DIR = paths.NEWSELA_ALIGNED + "features_alternative/"
# the directory to which all the numpy arrays will be stored
EMB_MODEL = paths.NEWSELA_ALIGNED + "model.bin"
# current model is trained with vectors of size 500, window = 5
# alpha = 0.01, min_alpha = 1.0e-9, negative sampling = 5. The third epoch is
# taken because it shows the best score on SimLex-999
EMB_SIZE = 500
N_ALTERNATIVES = 5
TOP_N = 20
# the number of substitution candidates to add (teh substitutions are chosen
# from the nearest word vecors that bear the same POS tag)
BINARY = lambda x: int(x)

"""
ORIGINAL_DATA = \
    paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt"
FEATURE_DIR = paths.NEWSELA_COMPLEX + "features/"
EMB_MODEL = paths.NEWSELA_COMPLEX + "model.bin"
EMB_SIZE = 500
BINARY = lambda x: 0 if int(x) <= 3  else 1 
"""


class CustomFeatureEstimator:

    tag_to_num = {'A': 5, 'MD': 2, 'PDT': 9, 'RP': 8, 'IN': 6, '-RRB-': 7,
                  'CC': 11, 'LS': 17, 'J': 3, 'SYM': 18, 'N': 1, 'P': 12,
                  'UH': 16, 'W': 15, 'V': 19, '-LRB-': 13, 'DT': 10, 'CD': 4,
                  'FW': 14, 'PHRASE': 0}
    num_to_tag = {0: 'PHRASE', 1: 'N', 2: 'MD', 3: 'J', 4: 'CD', 5: 'A',
                  6: 'IN', 7: '-RRB-', 8: 'RP', 9: 'PDT', 10: 'DT', 11: 'CC',
                  12: 'P', 13: '-LRB-', 14: 'FW', 15: 'W', 16: 'UH', 17: 'LS',
                  18: 'SYM', 19: 'V'}

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
        if N_ALTERNATIVES > 0:
            if feature_names[0] != 'wv' and (feature_names[1] != 'wv' or feature_names[0] != 'POS'):
                print('If alternatives are to be calculated, place POS and wv'
                      'in the beginning')
        for i in range(len(self.features)):
            for dependency in self.features[i]['dep']:
                # some features require that another feature is executed
                # beforehand (e.g. word vectors need to know the POS tags)
                if dependency not in [x['name'] for x in self.features[:i]]:
                    print("Feature " + dependency + " should come before feature "
                                                    + self.features[i]['name'])
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
            result.append(numpy.zeros(N_ALTERNATIVES + 1))
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
        list_of_words = [re.sub('[^a-zA-Z0-9\- \t,.!@#%&\*\(\)\'\";:?/]', '', x) for x in list_of_words]
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
            result.append(numpy.zeros(N_ALTERNATIVES + 1))
            for j in range(len(line['words'])):
                word = line['words'][j]
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
            result.append(numpy.zeros(N_ALTERNATIVES + 1))
            for j in range(len(line['words'])):
                word = line['words'][j]
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
        next_id = len(self.tag_to_num) + 1
        for tag in tags:
            if tag not in self.tag_to_num:
                self.tag_to_num[tag] = next_id
                self.num_to_tag[next_id] = tag
                next_id += 1
            result.append(numpy.array([self.tag_to_num[tag]]))
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
            tag = self.num_to_tag[self.results['POS'][i][0]]
            target = data[i]['words'][0] + '_' + tag
            result.append([])
            if target not in model.vocab:
                result[-1].append(numpy.zeros(EMB_SIZE * (N_ALTERNATIVES + 1) + N_ALTERNATIVES))
            else:
                if i% 100 == 0:
                    print(i)
                result[-1].append(model[target])
                if N_ALTERNATIVES > 0:
                    closest = model.wv.most_similar(positive=[target], topn=TOP_N)
                    j = 0
                    ind = 0
                    cosines = numpy.zeros(N_ALTERNATIVES)
                    while j < len(closest) and ind < N_ALTERNATIVES:
                        substitution, cosine = closest[j]
                        substitution = substitution.split('_')
                        word = '_'.join(substitution[:-1])
                        tag_check = substitution[-1]
                        if tag_check == tag:
                            cosines[ind] = cosine
                            file.write(re.sub('[^a-zA-Z0-9\- \t,.!@#%&\*\(\)\'\";:?/]', '', word) + '\t')
                            data[i]['words'].append(word)
                            data[i]['inds'].append(data[i]['inds'][0])
                            if word + '_' + tag in model.vocab:
                                result[-1].append(model[word + '_' + tag])
                            else:
                                result[-1].append(numpy.zeros(EMB_SIZE))
                            ind += 1
                        j += 1
                    file.write('\n')
                    while len(result[-1]) != N_ALTERNATIVES + 1:
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
        if N_ALTERNATIVES > 0:
            result = []
            for line in data:
                if line['subst'] in line['words']:
                    result.append(line['words'].index(line['subst']))
                else:
                    result.append(N_ALTERNATIVES + 1)
        return [line['score'] for line in data]

    def n_gram_feature(self, data, n):
        """
        Get google n-gram-score
        :param data:
        :param n:
        :return:
        """
        result = numpy.zeros((len(data), N_ALTERNATIVES + 1))
        dictionary = {}
        # dictionary of values that are to be looked up in the n-grams
        for i in range(len(data)):
            line = data[i]
            for j in range(len(line['inds'])):
                id = line['inds'][j]
                if id == -1 or id - n + 1 < 0:
                    continue
                query = (' '.join(line['sent'].split(' ')[id - n + 1: id]) + ' ' + line['words'][j]).lstrip(' ')
                query = re.sub('`', '\'', query)
                if query not in dictionary:
                    dictionary[query] = []
                dictionary[query].append((i, j))
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
            result.append(numpy.zeros((N_ALTERNATIVES + 1) * 2))
            # result[i][j * 2] is the count of token[j] in lines with
            # hit_id = data[i]['hit']
            # result[i][j * 2 + 1] is the number of such lines
            if data[i]['hit'] is None:
                continue
            tokens = data[i]['words'] + [data[i]['phrase']]
            # tokens for which the hit frequency is to be counted
            if data[i]['hit'] not in hits:
                hits[data[i]['hit']] = [i]
            else:
                hits[data[i]['hit']].append(i)
            for line_id in hits[data[i]['hit']]:
                for j in range(len(tokens)):
                    result[i][j * 2] += data[line_id]['sent'].count(tokens[j])
                    result[i][j * 2 + 1] += 1
                tmp = data[line_id]['words'] + [data[line_id]['phrase']]
                for j in range(len(tmp)):
                    result[line_id][j * 2] += data[i]['sent'].count(tmp[j])
                    result[line_id][j * 2 + 1] += 1
        return result


def get_raw_data():
    """
    Load the raw data (in Chris format) in memory
    :return:
    """
    with open(ORIGINAL_DATA) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    lines = [{'words': [x[0]], 'sent': x[3], 'inds': [int(x[1])],
              'score':BINARY(x[2]), 'phrase': [], 'hit': None,
              'subst': x[-1]} for x in lines]
    return lines


def get_cwi_data():
    """
    Load the data (in CWI Semeval format) into memory
    :return:
    """
    with open(CWI_DATA) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    for i in range(len(lines)):
        if len(lines[i][4].split(' ')) > 1:
            # the target is a phrase
            ind1 = int(len(lines[i][1][:int(lines[i][2])].split(' ')))
            ind2 = int(len(lines[i][1][:int(lines[i][3])].split(' ')))
            lines[i] = {'phrase': lines[i][4], 'words': [], 'hit': lines[i][0],
                        'sent': lines[i][1], 'score': int(lines[i][9]),
                        'inds': [], 'phrase_ind': [ind1, ind2]}
        else:
            # the target is a word
            ind = int(len(lines[i][1][:int(lines[i][2])].split(' ')))
            lines[i] = {'words': [lines[i][4]], 'phrase': None,
                        'hit': lines[i][0], 'sent': lines[i][1],
                        'score': int(lines[i][9]), 'inds': [ind],
                        'phrase_ind': []}
    return lines


if __name__ == "__main__":
    fe = CustomFeatureEstimator(["wv", "hit", "sent_syllab", "word_syllab",
                                "word_count", "mean_word_length",
                                "synset_count", "synonym_count", "labels",
                                 "1-gram", "2-gram", "3-gram", "4-gram",
                                 "5-gram"])
    # fe = CustomFeatureEstimator(["1-gram", "word_count"])
    # TODO: Average synsets and synonyms count and n-gram frequencies
    fe.calculate_features(get_raw_data())
    exit(0)
    features = fe.load_features()
    labels = fe.load_labels()
    data = get_raw_data()
    if len(data) != len(features):
        exit(-1)
    for i in range(len(data)):
        line = data[i]
        print('\n')
        print(line['sent'] + '\t' + str(line['words']) + '\t' + str(line['phrase']))
        # print("POS: " + CustomFeatureEstimator.num_to_tag[features[i][0]])
        print("hit: " + str(features[i][1:13]))
        print("sent_syllab: " + str(features[i][13]))
        print("word_syllab: " + str(features[i][14:20]))
        print("word_count: " + str(features[i][21]))
        print("mean_word_length: " + str(features[i][21]))
        print("synset_count: " + str(features[i][22:28]))
        print("synonym_count: " + str(features[i][28:34]))
        print("1-gram: " + str(features[i][34:40]))
        print("2-gram: " + str(features[i][40:46]))
        # print("3-gram: " + str(features[i][46:52]))
        # print("4-gram: " + str(features[i][52:58]))
        # print("5-gram: " + str(features[i][58:64]))
        print("cosines: " + str(features[i][-5:]))
        # print("wv: " + str(features[i][64:3064]))
        print("label: " + str(labels[i]))