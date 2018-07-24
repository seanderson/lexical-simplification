"""
Classifies words as complex or simple using methods based on Reno Kriz's
Simplification Using Paraphrases and Context-based Lexical Substitution
"""
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
import analyzer
import dataLoader
import generate_features
import copy
import random


TESTCLASSIFY = False
IMPORTDATA = True
LINEAR_REG_TEST = True
NNET = True
KERAS = True
GRIDSEARCH = False
DATA_TYPES = ['num', 'bi_num', 'bi_str', 'bi_arr']
DATA_IN_TYPE = 'bi_num'
DATA_USE_TYPE = 'bi_arr'
ALL_COMPLEX = False
REMOVE_ZEROS = False
UNIQUE_ONLY = False
DEBUG = False

USE_WORD_VECS = False
WORD_ONLY_CONFIG = [True, True, True, True, True, False, False, False, False, False, False, False, False]
CONTEXT_ONLY_CONFIG = [False, False, False, False, False, False, True, True, True, True, True, True, True]
ALL_FEATURES_CONFIG = [True, True, True, True, True, False, True, True, True, True, True, True, True]
NO_FEATURES = [False, False, False, False, False, False, False, False, False, False, False, False, False]
DENSITY_ONLY = [False, False, False, False, False, False, True, False, False, False, False, False, False]
CURRENT_CONFIG = ALL_FEATURES_CONFIG


def getStateAsString():
    s = ''
    s += 'NNET = ' + str(NNET) + '\n'
    s += 'DATA_IN_TYPE = ' + str(DATA_IN_TYPE) + '\n'
    s += 'DATA_USE_TYPE = ' + str(DATA_USE_TYPE) + '\n'
    s += 'ALL_COMPLEX = ' + str(ALL_COMPLEX) + '\n'
    s += 'REMOVE_ZEROS = ' + str(REMOVE_ZEROS) + '\n'
    s += 'DEBUG = ' + str(DEBUG) + '\n'
    return s


def convert_data(fromType, toType, data):
    if fromType == toType:
        return data
    if fromType == 'num':
        if DATA_USE_TYPE == 'bi_num':
            data = map(num_to_bi_num, data)
        elif DATA_USE_TYPE == 'bi_str':
            data = map(num_to_str, data)
        elif DATA_USE_TYPE == 'bi_arr':
            data = map(num_to_arr, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType ' + toType)
    elif fromType == 'bi_num':
        if DATA_USE_TYPE == 'num':
            data = map(bi_num_to_num, data)
        elif DATA_USE_TYPE == 'bi_str':
            data = map(bi_num_to_str, data)
        elif DATA_USE_TYPE == 'bi_arr':
            data = map(bi_num_to_arr, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType ' + toType)
    elif fromType == 'bi_str':
        if DATA_USE_TYPE == 'num':
            data = map(str_to_num, data)
        elif DATA_USE_TYPE == 'bi_num':
            data = map(str_to_bi_num, data)
        elif DATA_USE_TYPE == 'bi_arr':
            data = map(str_to_arr, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType ' + toType)
    elif fromType == 'bi_arr':
        if DATA_USE_TYPE == 'num':
            data = map(bi_arr_to_num, data)
        elif DATA_USE_TYPE == 'bi_num':
            data = map(bi_arr_to_bi_num, data)
        elif DATA_USE_TYPE == 'bi_str':
            data = map(bi_arr_to_str, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType ' + toType)
    else:
        print('PROBLEM CONVERTING DATA: no converter set up for fromType ' + fromType)
    return data


def num_to_bi_num(item):
    item = int(item)
    if item < 3:
        return 0
    else:
        return 1


def num_to_str(item):
    """
    Classifies item as either simple 's' or complex 'c'
    :param item: a number from 0-9
    :return: either 's' or 'c' depending on if item is less than 3
    """
    item = int(item)
    if item < 3:
        return 's'
    else:
        return 'c'


def num_to_arr(item):
    item = int(item)
    item = num_to_str(item)
    return str_to_arr(item)


def bi_num_to_num(num):
    num = int(num)
    if num == 0:
        return 0
    elif num == 1:
        return 9
    else:
        print('PROBLEM: num not 1 or 0')
        return -1


def bi_num_to_str(num):
    num = int(num)
    if num == 1:
        return 'c'
    elif num == 0:
        return 's'
    else:
        print('PROBLEM: num not 1 or 0')
        return '?'


def bi_num_to_arr(num):
    str = bi_num_to_str(num)
    return str_to_arr(str)


def bi_arr_to_num(arr):
    num = bi_arr_to_bi_num(arr)
    return bi_arr_to_num(num)


def bi_arr_to_bi_num(arr):
    if arr[0] == 1:
        return 1
    elif arr[1] == 1:
        return 0
    else:
        print('PROBLEM: arr not [0,1] or [1,0]')
        return -1


def bi_arr_to_str(arr):
    if arr[0] == 1:
        return 'c'
    elif arr[1] == 1:
        return 's'
    else:
        print('PROBLEM: arr not [0,1] or [1,0]')
        return '?'


def str_to_num(s):
    num = str_to_bi_num(s)
    return bi_num_to_num(num)


def str_to_bi_num(s):
    if s == 's':
        return 0
    elif s == 'c':
        return 1
    else:
        print('PROBLEM: Y label ' + str(s) + ' not s or c')
        return -1


def str_to_arr(s):
    if s == 's':
        return [0, 1]
    elif s == 'c':
        return [1, 0]
    else:
        print('PROBLEM: Y label ' + str(s) + ' not s or c')
        return [0, 0]


def prob_num_to_num(num):
    # TODO test
    num = int(round(num))
    possible = [0,1,2,3,4,5,6,7,8,9]
    if num not in possible:
        print('PROBLEM: prob_num_to_num expected to produce num in 1-9, got '+str(num))
    return num


def prob_bi_num_to_str(num):
    # TODO test
    if num > .5:
        return 'c'
    else:
        return 's'


def prob_arr_to_str(arr):
    if arr[0] > arr[1]:
        return 'c'
    else:
        return 's'


def five_fold_test(X, Y):
    """
    Scales data and does a five-fold test on it
    :param X: feature data
    :param Y: classifications
    :return: [predicted categorizations, actual categorizations]
    """
    print("Initializing Test")
    results = [[], []]
    if len(X) != len(Y):
        return -1
    if DEBUG:
        X = X[:200]
        Y = Y[:200]
    numTimesToTest = 5
    # shuffle data
    temp = list(zip(copy.copy(X), copy.copy(Y)))
    random.shuffle(temp)
    tempX, tempY = zip(*temp)
    available = [tempX, list(tempY)]
    # print(calc_num_in_categories(available[1]))

    # split into fifths
    n = len(available[0]) / numTimesToTest
    fifths = [[[],[]], [[],[]], [[],[]], [[],[]], [[],[]]]
    for i in range(numTimesToTest):
        fifths[i][0] = available[0][n*i:n*(i+1)]
        fifths[i][1] = available[1][n*i:n*(i+1)]

    for i in range(numTimesToTest):
        print("Testing: " + str(i) + " Out of " + str(numTimesToTest))
        test = fifths[i]
        train = [[],[]]
        for j in range(len(fifths)):
            if i != j:
                train[0] += fifths[j][0]
                train[1] += fifths[j][1]
        # standardize feature data
        scaler = preprocessing.StandardScaler()
        train = [scaler.fit_transform(np.asarray(train[0]).astype(np.float)),
                 train[1]]
        test = [scaler.transform(np.asarray(test[0]).astype(np.float)),
                test[1]]
        # Run
        if NNET:
            if not KERAS:
                clf = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', alpha=0, solver='adam', learning_rate='adaptive')
                clf.fit(train[0], train[1])
                preds = clf.predict(test[0])
            else:
                earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
                callbacks = [earlyStopping]
                callbacks = None
                train[0] = np.array(train[0])
                train[1] = np.array(train[1])
                clf = keras_NN(len(train[0][0]),(10,),.1)
                clf.fit(train[0], train[1], epochs=50, batch_size=128, verbose=2, validation_split=.01, callbacks=callbacks,shuffle=True)
                preds = clf.predict(test[0])
        else:
            clf = classify(train)
            preds = clf.predict(test[0])
        # results.append(calc_percent_right(test, preds))
        if DATA_USE_TYPE == 'bi_arr':
            preds = map(prob_arr_to_str,preds)
            intermediateType = 'bi_str'
        elif DATA_USE_TYPE == 'num':
            preds = map(prob_num_to_num, preds)
            intermediateType = 'num'
        elif DATA_USE_TYPE == 'bi_num':
            preds = map(prob_bi_num_to_str, preds)
            intermediateType = 'bi_str'
        else:
            intermediateType = DATA_USE_TYPE
        preds = convert_data(intermediateType,DATA_USE_TYPE, preds)
        results[0] = np.append(results[0], preds)
        results[1] = np.append(results[1], test[1])
    return results


def classify(data):
    """
    trains a SVM on data
    :param data: the data to train the SVM on. In format [X,Y]
    :return: the trained SVM
    """
    clf = svm.SVC(C=1000.0, cache_size=500, gamma=1, kernel='rbf')
    #clf = svm.SVC(kernel='brbf', C=1, verbose=False, probability=False, degree=3, shrinking=True, max_iter = -1, decision_function_shape='ovr', random_state=None, tol=0.001, cache_size=200, coef0=0.0, gamma=0.1, class_weight=None)
    clf.fit(data[0], data[1])
    return clf


def keras_NN(inDim, hiddenShape = (10,),learningRate = .001):
    adam = optimizers.adam(lr= learningRate)
    model = Sequential()
    for layer in range(len(hiddenShape)):
        if layer == 0:
            model.add(Dense(hiddenShape[layer], input_dim=inDim, kernel_initializer="uniform", activation="tanh"))
        else:
            model.add(Dense(hiddenShape[layer], kernel_initializer="uniform", activation="tanh"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    print('Making network')
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['acc'])
    return model


def grid_search(X, Y, cutoff=-1):
    folds = 5
    print('doing grid search')
    '''if(BINARY_CATEGORIZATION):
        for i in range(len(Y)):
            Y[i] = str_to_bin_category(Y[i])'''
    if cutoff > 0:
        temp = list(zip(X, Y))
        random.shuffle(temp)
        X, Y = zip(*temp)
        if cutoff>len(X):
            print('Warning: cutoff larger than data; cutoff: '+str(cutoff)+' len data: '+str(len(X)))
        else:
            X = X[:cutoff]
            Y = Y[:cutoff]
    sc = analyzer.Analyzer(DATA_USE_TYPE).getScorer()
    if NNET:
        if not KERAS:
            # hiddenLayerSizes = [(60,),(40,),(20,),(15,),(10,),(5,),(1,)]
            hiddenLayerSizes = [(300,150),(100,50,),(80,40,),(60,30,),(40,20,),(20,10,),(15,7,),(10,15,),(5,2,)]
            activations = ['identity', 'logistic', 'tanh', 'relu']
            solvers = ['lbfgs', 'sgd', 'adam']
            learningRates = ['constant', 'invscaling', 'adaptive']
            alphas = [.1, .001, .00001, .0000001]
            parameters = {'hidden_layer_sizes': hiddenLayerSizes, 'activation': ['tanh'], 'solver': ['adam'], 'learning_rate': ['adaptive'], 'alpha': [0], 'early_stopping':[True]}
            if DEBUG:
                parameters = {'hidden_layer_sizes': [(20,), (10,)]}
            evaluator = MLPClassifier()
            scorer = make_scorer(sc, labels=['c'], average=None)
        else:
            inDim = [len(X[0])]
            shapes = [(10,),(30,),(50,),(70,),(90,),(110,),(130,),(150,)]
            s = [(3000,),(4000,),(5000,)]
            shapes2L = [(1,1,),(100,50,),(500,250,),(1000,500,),(1500,750,),(2000,1000,),(2500,1250,),(3000,1500,),(4000,2000,),(5000,2500,)]
            shapes3L = [(1,1,1,),(100,50,25,),(500,250,125,),(1000,500,125,),(1500,750,375,),(2000,1000,500,),(2500,1250,625,),(3000,1500,750,),(4000,2000,1000,),(5000,2500,1250,)]
            shapes_weird_but_good = [(10,30,50,70,110,130,150)]
            lrs = [.001]
            parameters = {'inDim':inDim,'hiddenShape':shapes3L,'learningRate':lrs}
            evaluator = KerasClassifier(build_fn=keras_NN, epochs=100,verbose=2)
            scorer = make_scorer(sc, labels=['c'], average=None)
    else:
        #parameters = {'kernel': ['rbf'], 'C': [.01, .1, 1, 10, 100, 1000],
        #              'gamma': [.001,.01,.1,1,10,100,1000]}
        parameters = {'kernel': ['rbf'], 'C': [800, 900, 1000, 1100, 1200],
            'gamma': [.01, .5, 1, 5, 10]}
        if(DEBUG):
            parameters = {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [1, 10], 'early_stopping': [True]}
        evaluator = svm.SVC()
        scorer = make_scorer(sc, labels=['c'], average=None)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    clf = GridSearchCV(evaluator, parameters, scoring=scorer, verbose=3, n_jobs=1, cv=folds)
    clf.fit(X,Y)
    scores = clf.cv_results_
    return clf.best_score_, clf.best_estimator_.get_params(), scores


def analyzeScores(scores):
    scoresMean = scores['mean_test_score']
    scoresMean = np.array(scoresMean)
    return scoresMean


if __name__ == '__main__':
    a = analyzer.Analyzer(DATA_USE_TYPE)
    if TESTCLASSIFY:
        iris = datasets.load_iris()
        rawDat = five_fold_test(iris.data, iris.target)
        processedData = []
        for i in range(len(rawDat[0])):
            processedData.append([rawDat[0][i],rawDat[1][i]])
        print(a.calc_percent_right(processedData))
    if IMPORTDATA:
        fe = generate_features.CustomFeatureEstimator(["POS", "sent_syllab", "word_syllab",
                                     "word_count", "mean_word_length", "wv",
                                     "synset_count", "synonym_count", "labels"])
        # TODO: Average synsets and synonyms count and n-gram frequencies
        fe.calculate_features(generate_features.get_raw_data())
    if LINEAR_REG_TEST:
        fe = generate_features.CustomFeatureEstimator(
            ["POS", "sent_syllab", "word_syllab",
             "word_count", "mean_word_length", "wv",
             "synset_count", "synonym_count", "labels"])
        featureData = fe.load_features()
        complexScores = fe.load_labels()
        for labelInd in range(len(complexScores)):
            complexScores[labelInd] = num_to_str(
                complexScores[labelInd])
        model = LinearRegression()
        XTr, XTe, YTr, Yte = train_test_split(featureData,complexScores,test_size=.3)
        model.fit(XTr,YTr)
        preds = model.predict(XTe)
    if not TESTCLASSIFY:
        fe = generate_features.CustomFeatureEstimator(["POS", "sent_syllab", "word_syllab",
                                     "word_count", "mean_word_length", "wv",
                                     "synset_count", "synonym_count", "labels"])
        featureData = fe.load_features()
        complexScores = fe.load_labels()
        if UNIQUE_ONLY:
            featureData, complexScores = dataLoader.remove_duplicates(featureData,complexScores)
        for i in range(len(featureData)):
            featureData[i] = featureData[i][:-1]
        if REMOVE_ZEROS:
            tempX = []
            tempY = []
            for labelInd in range(len(complexScores)):
                if not (complexScores[labelInd] == 0 or complexScores[labelInd] == '0'):
                    tempX.append(complexScores[labelInd])
                    tempY.append(featureData[labelInd])
            featureData = tempY
            complexScores = tempX
        if DATA_IN_TYPE != DATA_USE_TYPE:
            featureData = convert_data(DATA_IN_TYPE,DATA_USE_TYPE, featureData)
        if GRIDSEARCH:
            bestScore, bestEst, scores = grid_search(featureData,complexScores,cutoff=10000)
            print(analyzeScores(scores))
            print(str(bestScore))
            print(bestEst)
        rawDat = five_fold_test(featureData, complexScores)
        featureData = None
        complexScores = None
        if ALL_COMPLEX:
            if DATA_USE_TYPE == 'bi_str':
                for i in range(len(rawDat[0])):
                    rawDat[0][i] = 'c'
            elif DATA_USE_TYPE == 'num':
                for i in range((len(rawDat[0]))):
                    rawDat[0][i] = 9
            elif DATA_USE_TYPE == 'bi_num':
                for i in range((len(rawDat[0]))):
                    rawDat[0][i] = 1
            elif DATA_USE_TYPE == 'bi_arr':
                for i in range((len(rawDat[0]))):
                    rawDat[0][i] = [1,0]
            else:
                print('ERROR: DATA_USE_TYPE '+DATA_USE_TYPE+' does not have an ALL_COMPLEX case')
        processedData = a.process_results(rawDat)
        rawDat = None
        precision = a.calc_precision(processedData)
        recall = a.calc_recall(processedData)
        print(getStateAsString())
        print('[simpleCorrect, simpleIncorrect, complexCorrect, complexIncorrect]:')
        print([len(category) for category in processedData])
        print('% categorically correct')
        print(a.calc_percent_categorically_right(processedData))
        print('(precision, recall, f_measure')
        print(precision, recall, a.calc_f_measure(precision, recall))
