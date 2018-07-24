from lexenstein.morphadorner import MorphAdornerToolkit  # for syllable count
import numpy
import statistics
import classpaths as paths
from nltk.corpus import wordnet as wn
from newsela_pos import *  # POS tagger
import gensim


ORIGINAL_DATA = paths.NEWSELA_ALIGNED + "dataset.txt"
# the data in "Chris" format, i.e. a line with tab-separated values:
# word  ind score   sentence    substituition (the latter is optional)
FEATURE_DIR = paths.NEWSELA_ALIGNED + "features"
# the directory to which all the numpy arrays will be stored
EMB_MODEL = paths.NEWSELA_ALIGNED + "model.bin"
# current model is trained with vectors of size 500, window = 5
# alpha = 0.01, min_alpha = 1.0e-9, negative sampling = 5. The third epoch is
# taken because it shows the best score on SimLex-999
EMB_SIZE = 500
BINARY = lambda x: int(x)

"""
ORIGINAL_DATA = \
    paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt"
FEATURE_DIR = paths.NEWSELA_COMPLEX + "features"
EMB_MODEL = paths.NEWSELA_COMPLEX + "model.bin"
EMB_SIZE = 500
BINARY = lambda x: 0 if int(x) <= 3  else 1 
"""


class CustomFeatureEstimator:

    tag_to_num = {'A': 5, 'MD': 2, 'PDT': 9, 'RP': 8, 'IN': 6, '-RRB-': 7,
                  'CC': 11, 'LS': 17, 'J': 3, 'SYM': 18, 'N': 1, 'P': 12,
                  'UH': 16, 'W': 15, 'V': 0, '-LRB-': 13, 'DT': 10, 'CD': 4,
                  'FW': 14}
    num_to_tag = {0: 'V', 1: 'N', 2: 'MD', 3: 'J', 4: 'CD', 5: 'A', 6: 'IN',
                  7: '-RRB-', 8: 'RP', 9: 'PDT', 10: 'DT', 11: 'CC', 12: 'P',
                  13: '-LRB-', 14: 'FW', 15: 'W', 16: 'UH', 17: 'LS', 18: 'SYM'}

    def __init__(self, feature_names):
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
        for i in range(len(self.features)):
            for dependency in self.features[i]['dep']:
                # some features require that another feature is executed
                # beforehand (e.g. word vectors need to know the POS tags)
                if dependency not in [x['name'] for x in self.features[:i]]:
                    print("Feature " + dependency + " should come before feature "
                                                    + self.features[i]['name'])
                    try:
                        self.results[dependency] = \
                            numpy.load(FEATURE_DIR + '/' + dependency + '.npy')
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
            "sent_syllab": {"func": self.sent_syllable_feature, "dep": []},
            "word_syllab": {"func": self.word_syllable_feature, "dep": []},
            "word_count": {"func": self.word_count_feature, "dep": []},
            "mean_word_length": {"func": self.mean_word_length_feature,
                                 "dep": []},
            "synset_count": {"func": self.synset_count_feature, "dep": []},
            "synonym_count": {"func": self.synonym_count_feature, "dep": []},
            "POS": {"func": self.pos_tag_feature, "dep": []},
            "labels": {"func": self.get_labels, "dep": []},
            "wv": {"func": self.word_embeddings_feature, "dep": ["POS"]}
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
            numpy.save(FEATURE_DIR + '/' + feature["name"],
                       numpy.array(self.results[feature["name"]]))

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
            result.append(float(sum(output[ind: ind+n_of_words[i]])) / n_of_words[i])
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
            result.append(numpy.array(output[ind: ind + n_of_words]))
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
        return [len(line['sent'].split(' ')) for line in data]

    def mean_word_length_feature(self, data):
        """
        Calculate the mean word length per sentence
        :param data: See the entry for calculate_features
        :return:
        """
        return [statistics.mean([len(x) for x in line['sent'].split(' ') if
                          re.match('.*[^a-zA-Z].*', x)]) for line in data]

    def synset_count_feature(self, data):
        """
        Calculate the numer of wordnet synsets for each target word
        :param data: See the entry for calculate_features
        :return:
        """
        dict = {}
        result = []
        for line in data:
            result.append([])
            for word in line['words']:
                if word not in dict:
                    try:
                        dict[word] = len(wn.synsets(word))
                    except UnicodeDecodeError:
                        dict[word] = 0
                result[-1].append(dict[word])
            result[-1] = numpy.array(result[-1])
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
            result.append([])
            for word in line['words']:
                if word not in dict:
                    try:
                        senses = wn.synsets(word)
                    except UnicodeDecodeError:
                        senses = []
                    dict[word] = 0
                    for sense in senses:
                        dict[word] += len(sense.lemmas())
                result[-1].append(dict[word])
            result[-1] = numpy.array(result[-1])
        return result

    def pos_tag_feature(self, data):
        """
        Find out the POS tag of each target word that is in the sentence
        :param data:  See the entry for calculate_features
        :return:
        """
        tags = get_tags(data)
        result = []
        next_id = len(self.tag_to_num)
        for tag in tags:
            if tag not in self.tag_to_num:
                self.tag_to_num[tag] = next_id
                self.num_to_tag[next_id] = tag
                next_id += 1
            result.append(self.tag_to_num[tag])
        print(self.tag_to_num)
        print(self.num_to_tag)
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
        for i in range(len(data)):
            tag = self.num_to_tag[self.results['POS'][i]]
            target = data[i]['words'][0] + '_' + tag
            if target not in model.vocab:
                result.append(numpy.zeros(EMB_SIZE))
            else:
                result.append(model[target])
            return result

    def get_labels(self, data):
        """
        Create a list of labels (scores)
        :param data:
        :return:
        """
        return [line['score'] for line in data]


def get_raw_data():
    """
    Load the raw data (in Chris format) in memory
    :return:
    """
    with open(ORIGINAL_DATA) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    lines = [{'words': [x[0]], 'sent': x[3], 'inds': [int(x[1])],
              'score':BINARY(x[2])} for x in lines]
    return lines


if __name__ == "__main__":
    fe = CustomFeatureEstimator(["word_count", "sent_syllab"])
    # TODO: Average synsets and synonyms count and n-gram frequencies
    fe.calculate_features(get_raw_data())