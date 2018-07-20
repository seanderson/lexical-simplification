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


def calc_num_in_categories(l):
    """
    counts frequesncy of occurrence in a list of ints
    :param l: list of ints
    :return: list of occurrence where l[i] = index
    """
    categories = []
    for num in l:
        while len(categories) < num:
            categories.append(0)
        categories[num] += 1
    return categories


def calc_percent_right(processedDataCategory):
    """
    calculates the % right from a list [predicted category, actual category]
    :param processedDataCategory:
    :return:
    """
    if len(processedDataCategory) == 0:
        return 0
    check = []
    for j in range(len(processedDataCategory)):
        check.append(processedDataCategory[j][0] == processedDataCategory[j][1])
    numRight = 0
    for i in check:
        if i:
            numRight += 1
    return float(numRight) / float(len(check))

def process_results(results):
    '''
    reformats results into a confusion matrix
    :param results: [predicted categorizations, actual categorizations]
    :return: confusion matrix split into one list
    '''
    if BINARY_EVALUATION:
        simpleCorrect = []
        simpleIncorrect = []
        complexCorrect = []
        complexIncorrect = []
        for i in range(len(results[0])):
            right = int(results[1][i])
            pred = int(results[0][i])
            if right < 3:
                if pred < 3:
                    simpleCorrect.append([pred, right])
                else:
                    simpleIncorrect.append([pred, right])
            else:
                if pred >= 3:
                    complexCorrect.append([pred, right])
                else:
                    complexIncorrect.append([pred, right])
        data = [simpleCorrect, simpleIncorrect, complexCorrect, complexIncorrect]
    else:
        pred = []
        actual = []
        for i in range(len(results[0])):
            actual.append(int(results[1][i]))
            pred.append(int(results[0][i]))
        data = confusion_matrix(actual, pred, [0,1,2,3,4,5,6,7,8,9])
    return data


'''          correct    incorrect       A v P > complex  simple  
    complex     TP          FN          complex  CC TP  CI FN
     simple     TN          FP           simple  SI FP  SC TN
    [TN, FP, TP, FN]
'''


def process_results_bin(results):
    """
    A version of process_results that uses 's' and 'c' rather than comparing
    the category to 3
    :param results: [predicted categorizations, actual categorizations]
    :return: confusion matrix split into one list
    """
    simpleCorrect = []
    simpleIncorrect = []
    complexCorrect = []
    complexIncorrect = []
    for i in range(len(results[0])):
        right = results[1][i]
        pred = results[0][i]
        if right == 's':
            if pred == 's':
                simpleCorrect.append([pred,right])
            else:
                simpleIncorrect.append([pred,right])
        else:
            if pred == 'c':
                complexCorrect.append([pred,right])
            else:
                complexIncorrect.append([pred,right])
    data = [simpleCorrect, simpleIncorrect, complexCorrect, complexIncorrect]
    return data

def calc_TP(pData):
    TP = 0
    for i in range(len(pData[0])):
        TP += pData[i][i]
    return TP


def calc_avg_percent_right(pData):
    avg = 0
    for i in range(len(pData)):
        avg += calc_percent_right(pData[i])
    avg /= i
    return avg


def calc_percent_categorically_right(pData):
    if BINARY_EVALUATION:
        return float(len(pData[0])+len(pData[2])) /\
           float(sum([len(pData[0]), len(pData[1]), len(pData[2]), len(pData[3])]))
    else:
        return 0


def calc_precision(pData):
    if BINARY_EVALUATION:
        TP = len(pData[2])
        FP = len(pData[1])
    if TP + FP == 0:
        return 0
    return float(TP)/float(TP+FP)


def calc_recall(pData):
    if BINARY_EVALUATION:
        TP = len(pData[2])
        FN = len(pData[3])
    if TP + FN == 0:
        return 0
    return float(TP)/float(TP+FN)


def calc_f_measure(precision, recall):
    if precision + recall == 0:
        return -1
    return 2*precision*recall/(precision + recall)


def custom_f1_scorer(y, y_pred, **kwargs):
    if BINARY_CATEGORIZATION:
        if KERAS:
            y = map(bi_arr_to_str, y)
            y_pred = map(bi_nums_to_str, y_pred)
        data = process_results_bin([y_pred,y])
    else:
        data = process_results([y_pred,y])
    precision = calc_recall(data)
    recall = calc_recall(data)
    return calc_f_measure(precision, recall)
