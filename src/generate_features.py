from lexenstein.morphadorner import MorphAdornerToolkit
import numpy
import statistics
import classpaths as paths
from nltk.corpus import wordnet as wn
from newsela_pos import *
import gensim


EMB_SIZE = 500
ORIGINAL_DATA = paths.NEWSELA_ALIGNED + "dataset.txt"
FEATURE_DIR = paths.NEWSELA_ALIGNED + "features"
EMB_MODEL = paths.NEWSELA_ALIGNED + "model.bin"
BINARY = lambda x: int(x)

"""
ORIGINAL_DATA = \
    paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt"
FEATURE_DIR = paths.NEWSELA_COMPLEX + "features"
EMB_MODEL = paths.NEWSELA_COMPLEX + "model.bin"
BINARY = lambda x: 0 if int(x) <= 3  else 1 
"""


class CustomFeatureEstimator:

    tag_to_num = {}
    num_to_tag = {}

    def __init__(self, feature_names):
        """
        Creates an instance of the FeatureEstimator class.
        """
        self.all_features = {
            "sent_syllab": {"func": self.sent_syllable_feature, "dep": ["word_count"]},
            "word_syllab": {"func": self.word_syllable_feature, "dep": []},
            "word_count": {"func": self.word_count_feature, "dep": []},
            "mean_word_length": {"func": self.mean_word_length_feature, "dep": []},
            "synset_count": {"func": self.synset_count_feature, "dep": []},
            "synonym_count": {"func": self.synonym_count_feature, "dep": []},
            "POS": {"func": self.pos_tag_feature, "dep": []}
        }
        for key in self.all_features:
            self.all_features[key]["name"] = key
        self.features = [self.all_features[name] for name in feature_names]
        for i in range(len(self.features)):
            for dependency in self.features[i]['dep']:
                if dependency not in [x['name'] for x in self.features[:i]]:
                    print("Feature " + dependency + " must come before feature "
                                                    + self.features[i]['name'])
                    exit(-1)
        self.results = {}

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
        for line in data:
            input += line['sent'].split(' ')
        output = self.syllabify(input)
        ind = 0
        result = []
        for i in range(len(data)):
            n_of_words = self.results['word_count'][i]  # N of words in sent
            result.append(float(sum(output[ind: ind+n_of_words])) / n_of_words)
            ind += n_of_words
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
        return result

    def word_embeddings_feature(self, data):
        """
        Get word embeddings from the default model
        :param data:  See the entry for calculate_features
        :return:
        """
        """result = []
        dict = {}  # a dictionary needed so that recalculating values is
        # not an issue
        model = gensim.models.KeyedVectors.load_word2vec_format(EMB_MODEL,
                                                                binary=True)
        for line in data:
            target = line['words'][0] + '_' + self.results['POS']
            if target in model.vocab:
            for word in line['words']:
                words = subst.strip().split(':')[1].strip()
                word_vector = numpy.zeros(size)
                for word in words.split(' '):
                        try:
                            word_vector = numpy.add(word_vector, model[words])
                        except KeyError:
                            pass
                    result.append(word_vector)
            for i in range(0, len(result)):
                result[i] = result[i].tolist()
            return result"""

    def get_word_vector(self, word, model, dict):
        """
        Look up the word in dict. If it is in dict, return the entry,
        if not, get the vector from the model,
        if this vector does not exist, return an empty numpy array
        :param word:
        :param model:
        :param dict:
        :return:
        """


def get_raw_data():
    """
    Load the raw data (in Chris format) in memory
    :return:
    """
    with open(ORIGINAL_DATA) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    lines = [{'words': [x[0]], 'sent': x[3], 'inds': [x[1]]}
             for x in lines]
    return lines


if __name__ == "__main__":
    fe = CustomFeatureEstimator(["word_count", "sent_syllab", "word_syllab",
                             "mean_word_length", "synset_count",
                             "synonym_count"])
    fe.calculate_features(get_raw_data())