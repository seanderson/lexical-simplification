from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
import statistics
import classpaths as paths


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

EMB_SIZE = 500
MORPH_ADORNER_TOOLKIT = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)


class CustomFeatureEstimator(FeatureEstimator):

    def calculateFeatures(self, corpus):
        data = []
        for line in corpus.split('\n'):
            line_data = line.strip().split('\t')
            data.append([line_data[0].strip(), line_data[1].strip(),
                         line_data[2].strip(),
                         '0:' + line_data[1].strip()])
            values = []
        for feature in self.features:
            values.append(feature[0].__call__(data, feature[1]))
        for i in range(len(values)):
            featureName = re.sub('[^a-zA-Z]', '_', self.identifiers[i][0])
            print(featureName)
            print(len(values[i]))
            # print(values[i])
            numpy.save(FEATURE_DIR + '/' + featureName, numpy.array(values[i]))
        return values

    def add_sent_syllable_feature(self, mat):
        """
        Adds a syllable count feature to the estimator. The value will be the
        average number of syllables in a word in this sentence.

        :param mat:         A configured MorphAdornerToolkit object.
        """
        self.features.append((self.sent_syllable_feature, [mat]))
        self.identifiers.append(('sent_syllab_count', None))

    def sent_syllable_feature(self, data, args):
        """
        A feature for assessing the average sentence syllable count
        :param data:
        :param args:
        :return:
        """
        mat = args[0]
        # Create the input for the Java application:
        input = []
        sent_length = []
        for line in data:
            input += line[0].split(' ')
            sent_length.append(len(line[0].split(' ')))

        # Run the syllable splitter:
        outr = mat.splitSyllables(input)

        # Decode output:
        out = []
        for o in outr:
            out.append(o.decode("latin1").replace(' ', '-'))

        # Calculate number of syllables
        result = [0]
        count = 0  # number of words in this sentence
        s_ind = 0  # id of the sentence to be processed
        for instance in out:
            if len(instance.strip()) > 0:
                count += 1
                if count > sent_length[s_ind]:
                    result[-1] /= (count - 1)
                    count = 1
                    result.append(0)
                result[-1] += len(instance.split('-'))
        result[-1] /= count
        return result

    def add_sent_length_feature(self):
        """
        Add a feature that calculates the number of words in each sentence
        :return:
        """
        function = lambda data, _: [len(line[0].split(' ')) for line in data]
        self.features.append((function, []))
        self.identifiers.append(('Sent_Length', None))

    def add_avg_sent_word_length_feature(self):
        self.features.append((lambda data, _: [statistics.mean(
            [len(x) for x in line[0].split(' ') if re.match('.*[^a-zA-Z].*', x)]
        ) for line in data], []))
        self.identifiers.append(('Avg_Word_Length', None))


def cwictorify():
    """
    Converts the original data from the Chris format to CWICTOR format used
    by lexenstein.
    CWICTOR format: sent    word    ind    score (binary)
    :return: the list of lines in CWICTOR format
    """
    lines = get_raw_data()
    return '\n'.join(['\t'.join([x['sent'], x['word'], x['ind'],
                                 str(BINARY(x['score']))]) for x in lines])


def get_raw_data():
    """
    Load the raw data (in Chris format) in memory
    :return: list of [word, word_index, complexity, sentence]
    """
    with open(ORIGINAL_DATA) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    lines = [{'word': x[0], 'sent': x[3], 'ind': x[1], 'score':x[2]}
             for x in lines]
    return lines


fe = CustomFeatureEstimator()
# fe.addLengthFeature('Complexity')  # word length (in chars)
# fe.addSynonymCountFeature('Simplicity')  # number of synonyms of a word
# fe.addSenseCountFeature('Simplicity')
# fe.add_avg_sent_word_length_feature()
fe.addWordVectorValues(EMB_MODEL, EMB_SIZE, 'Simplicity')  # embeddings
fe.addSyllableFeature(MORPH_ADORNER_TOOLKIT, 'Simplicity')  # word syllable count
# fe.add_sent_syllable_feature(MORPH_ADORNER_TOOLKIT)  # average sent syllable
# fe.add_sent_length_feature()
fe.calculateFeatures(cwictorify())