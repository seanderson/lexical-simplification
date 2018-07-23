import numpy
from lexenstein.identifiers import *
from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
import classpaths as paths
from nltk.corpus import wordnet
from nltk.corpus import cmudict
from keras.optimizers import adam
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import cast
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import  MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import copy
import random
import generate_features


def read_features(filepath, featureConfig=-1):
    """
    Reads features from a file created with the save() function
    :param filepath: the path to the file to read features from
    :param featureConfig: an array of booleans with length of data indicating
    whether to read that feature
    :return: a list of the features
    """
    data = []
    with open(filepath) as file:
        lines = file.readlines()
    for line in lines:
        data.append(line.split('\t')[3:])
    data = data[1:]
    if DEBUG:
        data = data[:100]
    # if not USE_WORD_VECS:
    #     for line in range(len(data)):
    #         data[line] = data[line][:-1300]
    if featureConfig == -1:
        return data
    for featureSetInd in range(len(data)):
        featureSet = data[featureSetInd]
        featureSet[-1] = featureSet[-1].rstrip('\n')
        featureInd = 0
        #while featureInd < len(data[featureSetInd]):
        while featureInd < len(featureConfig):
            if not featureConfig[featureInd]:
                # print(featureInd, len(data[featureSetInd]))
                featureSet.remove(data[featureSetInd][featureInd])
            featureInd += 1
        data[featureSetInd] = featureSet
    linesToRemove = []
    for lineInd in range(len(data)):
        if len(data[lineInd]) == 0:
            linesToRemove.append(data[lineInd])
    for ind in linesToRemove:
        data.remove(ind)
    '''for i in range(len(data)):
        data[i] = data[i][:-1]'''
    return data


def remove_duplicates(X,Y):
    linesToRemove = []
    words = {}
    for lineInd in range(len(X)):
        if X[lineInd][-1] in words:
            words[X[lineInd][-1]].append(lineInd)
        else:
            words[X[lineInd][-1]] = [lineInd]
    for word in list(words):
        if len(words[word]) > 1:
            for instance in words[word]:
                linesToRemove.append([X[instance],Y[instance]])
    for line in linesToRemove:
        X.remove(line[0])
        Y.remove(line[1])
    return X,Y


def read_complexities(filepath):
    """
    reads Kriz complexity scores from a file at filepath
    :param filepath:
    :return: a list of complexity scores from the file at filepath
    """
    complexities = []
    with open(filepath) as file:
        lines = file.readlines()
    for line in lines:
        complexities.append(line.split('\t')[2])
    if DEBUG:
        complexities = complexities[:100]
    return complexities


def read_from_new():
    files = ["word_count", "sent_syllab", "word_syllab",
             "mean_word_length", "synset_count",
             "synonym_count"]
    features = []
    lines = []
    path = generate_features.FEATURE_DIR
    for file in files:
        features.append(numpy.load(path + '/' + file + '.npy'))
    for i in range(len(features[0])):
        l = []
        for featureInd in range(len(features)):
            l.append(features[featureInd][i])
        lines.append(l)
    return lines


if __name__ == '__main__':
    feats = read_from_new()
    for feat in feats:
        print(feat)