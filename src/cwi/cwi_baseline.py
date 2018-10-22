import copy

import src.cwi.generate_features as gf
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

USE_WORD_VECTORS = True

CL_TYPES = ["AdaBoost", "Nearest Centroid", "Linear Regression", "Neural Network"]
CL_TYPE = CL_TYPES[3]

DATASETS = ["News_Dev.tsv", "News_Train.tsv", "News_Test.tsv",
         "WikiNews_Dev.tsv", "WikiNews_Train.tsv", "WikiNews_Test.tsv",
         "Wikipedia_Dev.tsv", "Wikipedia_Train.tsv", "Wikipedia_Test.tsv"]
# the list of filenames with different datasets
FEATURES_TO_USE = ["is_a_phrase", "word_length", "vowel_count", "word_syllab", "punctuation",
        "1-gram", "5-gram", "POS", "2-gram", "3-gram", "4-gram",
        "sent_syllab", "word_count", "hit", "lexicon", "mean_word_length"]
# Note that "wv" should always come the last and "is_a_phrase" must always
# be the first in teh list
if USE_WORD_VECTORS:
    FEATURES_TO_USE += ["wv_cosines", "wv"]


def train_classifier(x, y):
    """
    Train a classifier
    :param x: features to train on
    :param y: labels to train on
    :return:  created classifier
    """
    if CL_TYPE == "Neural Network":
        clf = MLPClassifier(hidden_layer_sizes=(100, 10), early_stopping=True, learning_rate_init=0.0001)
    elif CL_TYPE == "Linear Regression":
        clf = LogisticRegression(multi_class='ovr')
    elif CL_TYPE == "AdaBoost":
        depth = 3
        n_estimators=500
        print(depth, n_estimators)
        clf = AdaBoostClassifier(
            RandomForestClassifier(max_depth=depth, max_features=None),
            n_estimators=n_estimators, learning_rate=1)
    elif CL_TYPE == "Nearest Centroid":
        clf = NearestCentroid(shrink_threshold=1.)
    else:
        print("Unknown classifier type")
        exit(-1)
    clf.fit(x, y)
    return clf


def test_classifier(clf, x, y, multiword, clf_phrase=None):
    """
    Test a classifier
    :param clf:       the classifier to test
    :param x:         testing set features
    :param y:         testing set labels
    :param multiword: an array that stores 1, if the target sample is a phrase
    and 0 otherwise.
    :param clf_phrase: classifier to use for phrases
    :return: None
    """
    pred = clf.predict(x)
    for i in range(len(multiword)):
        if multiword[i] == 1:
            if clf_phrase is None:
                pred[i] = 1
            else:
                pred[i] = clf_phrase.predict([x[i]])

    precision, recall, fscore, _ = precision_recall_fscore_support(y, pred, average='macro')
    fscore_verified = f1_score(y, pred, average='macro')
    print("Precision: " + str(round(precision, 4)) +
          ", Recall: " + str(round(recall, 4)) +
          ", F-Score: " + str(round(fscore, 4)))
    if fscore != fscore_verified:
        print("Note that for some reason there is a difference in F-scores. "
              "The other possible value is: " + str(round(fscore_verified, 4)))


def load_dataset(num):
    """
    Load the data from a particular dataset into memory
    :param num: the index of the dataset in the DATASETS array
    :return:    features, labels
    """
    f = DATASETS[num]
    root = "/home/nlp/wpred/datasets/cwi/"
    if (num + 1) % 3 == 0:
        # if this is part of the testing set
        feature_file = root + "testset/english/" + f
        feature_dir = root + "testset/english/" + f[:-4] + "_Features/"
    else:
        feature_file = root + "traindevset/english/" + f
        feature_dir = root + "traindevset/english/" + f[:-4] + "_Features/"
    data = gf.get_cwi_data(feature_file)
    fe = gf.CustomFeatureEstimator(FEATURES_TO_USE, feature_dir)
    features = fe.load_features(phrase_features=True, data=data)
    labels = fe.load_labels()
    return features, labels


def fix_data(features, labels, scaler=None):
    """
    PRECONDITION: Word Vectors should come last in the list of features
    Fix up details of the features:
    0. Note all multi-word targets.
    1. z-score scale all features except for the wordvectors
    :param features: features to scale
    :param labels:   labels that correspond to these features
    :param scaler:   scaler to use to scale the features.
                     Create a new on, if scaler = None
    :return: scaler, features, labels, features_for_one_word_targets_only,
    labels_for_one_word_targets_only, features_phrase, labels_phrase, is_a_phrase?_array
    """
    ismultiword = [f[0] for f in features]
    multi_word = np.array([i for i in range(len(ismultiword)) if ismultiword[i] == 1])
    one_word = np.array(
        [i for i in range(len(ismultiword)) if ismultiword[i] == 0])
    features = [f[1:] for f in features]
    if USE_WORD_VECTORS:
        # word vectors should not be scaled
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit([f[:-500] for f in features])
        oldfeatures = copy.deepcopy(features)
        features = scaler.transform([f[:-500] for f in oldfeatures])
        features = [np.concatenate((features[i], oldfeatures[i][-500:])) for i in range(len(features))]
    else:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(features)
        features = scaler.transform(features)
    oneword_features = np.delete(features, multi_word, 0)
    oneword_labels = np.delete(labels, multi_word, 0)
    multi_features = np.delete(features, one_word, 0)
    multi_labels = np.delete(labels, one_word, 0)
    return scaler, features, labels, oneword_features, oneword_labels, multi_features, multi_labels, ismultiword


def main():
    """
    Evaluate the performance of a model/features on different datasets
    :return:
    """
    print("Classifier used: " + CL_TYPE)
    if FEATURES_TO_USE[0] != "is_a_phrase":
        print("Word length feature should always be the first in the feature list")
        exit(-1)
    print("Are word vectors used? - " + str(USE_WORD_VECTORS))
    print("All the features used: " + str(FEATURES_TO_USE))

    for idx in [0, 3, 6]:
        print("Processing Dataset: " + ', '.join(DATASETS[idx: idx + 3]))

        features, labels = load_dataset(idx+1)
        features_dev, labels_dev = load_dataset(idx+0)

        print("Shapes of the features: " + str(features.shape)
              + " (training), " + str(features_dev.shape) + " (dev)")
        features = np.concatenate((features, features_dev))
        labels = np.concatenate((labels, labels_dev))

        scaler, features, labels, oneword_features, oneword_labels, phrase_features, phrase_labels, _ = \
            fix_data(features, labels, None)

        print("Num removed multi-word samples %d" % (len(features)-len(oneword_features)))

        print("Trainig common classifier: ")
        clf = train_classifier(features, labels)
        print("Trainig word-specific classifier: ")
        clf_oneword = train_classifier(oneword_features, oneword_labels)
        print("Trainig phrase-specific classifier: ")
        clf_phrase = train_classifier(phrase_features, phrase_labels)

        # testing data
        features_test, labels_test = load_dataset(idx+2)
        _, features_test, labels_test, oneword_features, _, _, _, is_multi_word = \
            fix_data(features_test, labels_test, scaler)

        print("Num targets %d" % len(features_test))
        print("Num removed multi-word targets %d"% (len(features_test)-len(oneword_features)))

        print("Testing scores when marking all phrases as complex: ")
        test_classifier(clf_oneword, features_test, labels_test, is_multi_word, clf_phrase=None)
        print("Testing scores using a common classifier: ")
        test_classifier(clf, features_test, labels_test, is_multi_word, clf_phrase=clf)
        print("Testing scores using phrase classifier to predict phrases: ")
        test_classifier(clf_oneword, features_test, labels_test, is_multi_word,
                        clf_phrase=clf_phrase)


if __name__ == "__main__":
    main()
