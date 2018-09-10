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
from keras.callbacks import ModelCheckpoint
import keras.initializers
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
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import analyzer
import generate_features
import copy
import random

SCALE = False
TESTCLASSIFY = False
IMPORTDATA = False
LINEAR_REG_TEST = False
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

WORD_ONLY_CONFIG = ["POS","word_syllab","synset_count", "synonym_count","vowel_count"]

CONTEXT_CONFIG = ["POS","word_syllab","synset_count", "synonym_count", "vowel_count",
                  "sent_syllab","word_count","mean_word_length","1-gram", "2-gram",
                  "3-gram","4-gram","5-gram"]

WITH_VECS_CONFIG = ["POS","word_syllab","synset_count", "synonym_count","vowel_count",
                    "sent_syllab","word_count","mean_word_length","1-gram", "2-gram",
                    "3-gram","4-gram","5-gram", "wv"]

WITH_HIT_CONFIG = ["POS", "sent_syllab", "word_syllab", "word_count","vowel_count",
                   "mean_word_length", "synset_count", "synonym_count", "1-gram",
                   "2-gram", "3-gram","4-gram","5-gram","hit","wv","labels"]

ALL_FEATURES_CONFIG = ["POS", "sent_syllab", "word_syllab", "word_count","vowel_count", "mean_word_length",
                       "synset_count", "synonym_count", "1-gram", "2-gram",
                       "3-gram","4-gram","5-gram","hit","wv", "lexicon","labels"]

CONFIGS = [WORD_ONLY_CONFIG,CONTEXT_CONFIG,WITH_VECS_CONFIG,WITH_HIT_CONFIG,ALL_FEATURES_CONFIG]
ONLY_VECS = ["wv"]
CURRENT_CONFIG = ALL_FEATURES_CONFIG


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
        if toType == 'bi_num':
            data = map(num_to_bi_num, data)
        elif toType == 'bi_str':
            data = map(num_to_str, data)
        elif toType == 'bi_arr':
            data = map(num_to_arr, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType '+toType+' from type '+fromType)
    elif fromType == 'bi_num':
        if toType == 'num':
            data = map(bi_num_to_num, data)
        elif toType == 'bi_str':
            data = map(bi_num_to_str, data)
        elif toType == 'bi_arr':
            data = map(bi_num_to_arr, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType ' +toType+' from type '+fromType)
    elif fromType == 'bi_str':
        if toType == 'num':
            data = map(str_to_num, data)
        elif toType == 'bi_num':
            data = map(str_to_bi_num, data)
        elif toType == 'bi_arr':
            data = map(str_to_arr, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType '+ toType+' from type '+fromType)
    elif fromType == 'bi_arr':
        if toType == 'num':
            data = map(bi_arr_to_num, data)
        elif toType == 'bi_num':
            data = map(bi_arr_to_bi_num, data)
        elif toType == 'bi_str':
            data = map(bi_arr_to_str, data)
        else:
            print('PROBLEM CONVERTING DATA: no converter set up for toType ' +toType+' from type '+fromType)
    else:
        print('PROBLEM CONVERTING DATA: no converter set up for fromType ' + fromType)
    return data


def remove_zeros(X,Y):
    tempX = []
    tempY = []
    for labelInd in range(len(Y)):
        if not (Y[labelInd] == 0 or Y[labelInd] == '0'):
            tempX.append(Y[labelInd])
            tempY.append(X[labelInd])
    return tempX,tempY


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
    return bi_num_to_num(num)


def bi_arr_to_bi_num(arr):
    if arr[0] == 1:
        return 1
    elif arr[1] == 1:
        return 0
    else:
        print('PROBLEM: arr not [0,1] or [1,0]')
        return -1


def bi_arr_to_str(arr):
    if len(list(arr)) != 2:
        print('PROBLEM: arr not [0,1] or [1,0]')
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

    temp = None
    tempX = None
    tempY = None
    # split into fifths
    n = len(available[0]) / numTimesToTest
    fifths = [[[],[]], [[],[]], [[],[]], [[],[]], [[],[]]]
    for i in range(numTimesToTest):
        fifths[i][0] = available[0][n*i:n*(i+1)]
        fifths[i][1] = available[1][n*i:n*(i+1)]

    available = None

    for i in range(numTimesToTest):
        print("Testing: " + str(i+1) + " Out of " + str(numTimesToTest))
        test = fifths[i]
        train = [[],[]]
        for j in range(len(fifths)):
            if i != j:
                train[0] += fifths[j][0]
                train[1] += fifths[j][1]
        # standardize feature data
        if not SCALE:
            z = 0
            train = [np.asarray(train[0]).astype(np.float),train[1]]
            test = [np.asarray(test[0]).astype(np.float), test[1]]
        else:
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
                #callbacks = None
                train[0] = np.array(train[0])
                train[1] = np.array(train[1])
                clf = keras_NN(len(train[0][0]),(3000, 1500, 750),.00001)
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
        results[0] = results[0] + preds
        results[1] = results[1] + test[1]
    return results


def test_of_a_test(X, Y):
    shape = (1000,100)
    with open('networks/shape.txt','w') as file:
        file.write(str(shape)+'\n'+str(len(X[0])))
    #path = 'networks/saved-network-{epoch:02d}-{val_loss:.2f}.hdf5'
    path = 'networks/saved-network.hdf5'
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(filepath=path, monitor='val_loss',verbose=1,save_best_only=True)
    callbacks = [earlyStopping, checkpoint]
    #callbacks = [earlyStopping]
    evaluator = KerasClassifier(build_fn=keras_NN, inDim=len(X[0]), hiddenShape=shape, epochs=50, verbose=2)
    scorer = make_scorer(analyzer.custom_f1_scorer)
    params = {'callbacks':callbacks,'validation_split':.01}
    scores = cross_val_score(evaluator,X,Y,scoring=scorer, cv=5, verbose=3, fit_params=params)
    return scores


def please_work(X,Y):
    temp = list(zip(X, Y))
    random.shuffle(temp)
    X, Y = zip(*temp)
    per = int(len(X)*.60)
    X = numpy.asarray(X)
    Y = numpy.asarray(Y)
    train = [X[:per],Y[:per]]
    test = [X[per:],Y[per:]]
    clf = keras_NN(inDim=len(train[0][0]), hiddenShape=(3000, 1500, 750), learningRate=.00001)
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')]
    #callbacks = None
    clf.fit(train[0], train[1], epochs=50, batch_size=128, verbose=2, validation_split=.1, callbacks=callbacks, shuffle=False)
    preds = clf.predict(test[0])
    if DATA_USE_TYPE == 'bi_arr':
        preds = map(prob_arr_to_str, preds)
        intermediateType = 'bi_str'
    elif DATA_USE_TYPE == 'num':
        preds = map(prob_num_to_num, preds)
        intermediateType = 'num'
    elif DATA_USE_TYPE == 'bi_num':
        preds = map(prob_bi_num_to_str, preds)
        intermediateType = 'bi_str'
    else:
        intermediateType = DATA_USE_TYPE
    preds = convert_data(intermediateType, DATA_USE_TYPE, preds)
    return [preds, test[1]]


def please_work_sk(X, Y):
    temp = list(zip(X, Y))
    random.shuffle(temp)
    X, Y = zip(*temp)
    per = int(len(X) * .60)
    X = numpy.asarray(X)
    Y = numpy.asarray(Y)
    train = [X[:per], Y[:per]]
    test = [X[per:], Y[per:]]
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                                  verbose=1, mode='auto')
    callbacks = [earlyStopping]
    # callbacks = None
    evaluator = KerasClassifier(build_fn=keras_NN, inDim=len(X[0]), epochs=50,
                                verbose=2)
    evaluator.fit(train[0],train[1],callbacks=callbacks,validation_split=.1)
    preds = evaluator.predict(test[0])
    for i in range(len(preds)):
        if preds[i] == 1:
            preds[i] = 0
        else:
            preds[i] = 1
    preds = convert_data('bi_num','bi_arr',preds)
    return [preds,test[1]]


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


def keras_NN(inDim, hiddenShape = (10,),learningRate = .0001):
    adam = optimizers.adam(lr= learningRate)
    model = Sequential()
    bInitializer = keras.initializers.RandomUniform()
    bInitializer = keras.initializers.Zeros()
    kInitializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',distribution='normal')
    for layer in range(len(hiddenShape)):
        if layer == 0:
            model.add(Dense(hiddenShape[layer], input_dim=inDim, bias_initializer=bInitializer, kernel_initializer=kInitializer, activation="tanh"))
        else:
            model.add(Dense(hiddenShape[layer], kernel_initializer=kInitializer, bias_initializer=bInitializer, activation="tanh"))
    model.add(Dense(2,kernel_initializer=kInitializer,bias_initializer=bInitializer,activation='softmax'))
    #model.add(Activation("softmax"))
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
    a = analyzer.Analyzer(DATA_USE_TYPE)
    if SCALE:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = numpy.asarray(X)
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
            scorer = make_scorer(a.getScorer(), labels=['c'], average=None)
            clf = GridSearchCV(evaluator, parameters, scoring=scorer, verbose=3,
                               n_jobs=1, cv=folds)
            clf.fit(X, Y)
        else:
            earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
            callbacks = [earlyStopping]
            inDim = [len(X[0])]
            shapes = [(10,),(30,),(50,),(70,),(90,),(110,),(130,),(150,)]
            s = [(10,),(100,),(1000,),(100,10),(1000,100)]
            s1 = [(2000,),(3000,),(4000,),(5000,)]
            shapes2L = [(1,1,),(100,50,),(500,250,),(1000,500,),(1500,750,),(2000,1000,),(2500,1250,),(3000,1500,),(4000,2000,),(5000,2500,)]
            shapes3L = [(1,1,1,),(100,50,25,),(500,250,125,),(1000,500,125,),(1500,750,375,),(2000,1000,500,),(2500,1250,625,),(3000,1500,750,),(4000,2000,1000,),(5000,2500,1250,)]
            shapes_weird_but_good = [(10,30,50,70,110,130,150)]
            hiddenshape = s1
            if DEBUG:
                hiddenshape = [(100,)]
            lrs = [.0001]
            parameters = {'inDim':inDim,'hiddenShape':hiddenshape,'learningRate':lrs}
            evaluator = KerasClassifier(build_fn=keras_NN, epochs=100,verbose=2)
            scorer = make_scorer(analyzer.custom_f1_scorer, labels=['c'], average=None)
            clf = GridSearchCV(evaluator, parameters, scoring=scorer, verbose=3,
                               n_jobs=1, cv=folds)
            clf.fit(X, Y, callbacks=callbacks, validation_split=.1)

    else:
        # parameters = {'kernel': ['rbf'], 'C': [.01, .1, 1, 10, 100, 1000],
        #              'gamma': [.001,.01,.1,1,10,100,1000]}
        parameters = {'kernel': ['rbf'], 'C': [800, 900, 1000, 1100, 1200],
            'gamma': [.01, .5, 1, 5, 10]}
        if(DEBUG):
            parameters = {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [1, 10], 'early_stopping': [True]}
        evaluator = svm.SVC()
        scorer = make_scorer(analyzer.custom_f1_scorer)
        clf = GridSearchCV(evaluator, parameters, scoring=scorer, verbose=3,
                           n_jobs=1, cv=folds)
        clf.fit(X, Y)
    scores = clf.cv_results_
    return clf.best_score_, clf.best_estimator_.get_params(), scores


def analyzeScores(scores):
    scoresMean = scores['mean_test_score']
    scoresMean = np.array(scoresMean)
    return scoresMean


def test_over_multiple_feature_sets(featureSets, trainPath=None, testPath=None):
    scores = {}
    oneSet = testPath is None
    if trainPath is None:
        trainPath = generate_features.FEATURE_DIR
    a = analyzer.Analyzer(DATA_USE_TYPE)
    for i in range(len(featureSets)):
        featSet = featureSets[i]
        print('testing '+str(i)+' of '+str(len(featureSets))+': '+str(featSet))
        fe = generate_features.CustomFeatureEstimator(featSet,trainPath)
        featureData = fe.load_features()
        complexScores = fe.load_labels()
        if UNIQUE_ONLY:
            featureData, complexScores = remove_duplicates(featureData,complexScores)
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
            complexScores = convert_data(DATA_IN_TYPE,DATA_USE_TYPE, complexScores)
        if oneSet:
            XTr, XTe, YTr, YTe = train_test_split(featureData,complexScores, test_size=.20)
        else:
            XTr = featureData
            YTr = complexScores
        if NNET:
            test_of_a_test(XTr, YTr)
        else:
            five_fold_test(XTr, YTr)
        if not oneSet:
            fe = generate_features.CustomFeatureEstimator(featSet, testPath)
            XTe = fe.load_features()
            YTe = fe.load_labels()
        model = keras.models.load_model('networks/saved-network.hdf5')
        preds = model.predict(XTe)
        preds = map(prob_arr_to_str,preds)
        preds = convert_data('bi_str','bi_num',preds)
        YTe = convert_data('bi_arr','bi_num', YTe)
        result = classification_report(YTe,preds)
        scores[i] = [featSet,result]
    return scores


def grid_search_over_multiple_feature_sets(featureSets):
    scores = {}
    for i in range(len(featureSets)):
        featSet = featureSets[i]
        print('testing ' + str(i) + ' of ' + str(len(featureSets)) + ': ' + str(featSet))
        fe = generate_features.CustomFeatureEstimator(featSet)
        featureData = fe.load_features()
        complexScores = fe.load_labels()
        if UNIQUE_ONLY:
            featureData, complexScores = remove_duplicates(featureData, complexScores)
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
            complexScores = convert_data(DATA_IN_TYPE, DATA_USE_TYPE,
                                         complexScores)
        bestScore,bestEst,scoresG = grid_search(featureData,complexScores,cutoff=30000)
        scores[i] = [featSet, bestScore, bestEst, scoresG]
    return scores


if __name__ == '__main__':
    te = test_over_multiple_feature_sets(CONFIGS)
    for key in te.keys():
        print(te[key][0])
        print(te[key][1])
        print('')
    print('Done :)')
    #print(test_over_multiple_feature_sets([ALL_FEATURES_CONFIG],))
    if DEBUG:
        CONFIGS = [WORD_ONLY_CONFIG,CONTEXT_CONFIG]
    #results = grid_search_over_multiple_feature_sets(CONFIGS)
    results = grid_search_over_multiple_feature_sets([ALL_FEATURES_CONFIG])
    for key in results.keys():
        r = results[key]
        print(r[0])
        print(r[1])
        print(r[2])
        print('')
    a = analyzer.Analyzer(DATA_USE_TYPE)
    if TESTCLASSIFY:
        iris = datasets.load_iris()
        rawDat = five_fold_test(iris.data, iris.target)
        processedData = []
        for i in range(len(rawDat[0])):
            processedData.append([rawDat[0][i],rawDat[1][i]])
        print(a.calc_percent_right(processedData))
    if IMPORTDATA:
        fe = generate_features.CustomFeatureEstimator(ALL_FEATURES_CONFIG)
        # TODO: Average synsets and synonyms count and n-gram frequencies
        fe.calculate_features(generate_features.get_raw_data())
    if LINEAR_REG_TEST:
        fe = generate_features.CustomFeatureEstimator(CURRENT_CONFIG)
        featureData = fe.load_features()
        complexScores = fe.load_labels()
        for labelInd in range(len(complexScores)):
            complexScores[labelInd] = num_to_str(
                complexScores[labelInd])
        model = LinearRegression()
        XTr, XTe, YTr, Yte = train_test_split(featureData,complexScores,test_size=.3)
        model.fit(XTr,YTr)
        preds = model.predict(XTe)
    if not TESTCLASSIFY or LINEAR_REG_TEST:
        fe = generate_features.CustomFeatureEstimator(["POS", "sent_syllab", "word_syllab",
                                     "word_count", "mean_word_length", "wv",
                                     "synset_count", "synonym_count", "labels"],generate_features.FEATURE_DIR)
        #featureData = fe.load_features()
        complexScores = fe.load_labels()
        if UNIQUE_ONLY:
            featureData, complexScores = remove_duplicates(featureData,complexScores)
        if REMOVE_ZEROS:
            featureData, complexScores = remove_zeros(featureData, complexScores)
        if DATA_IN_TYPE != DATA_USE_TYPE:
            complexScores = convert_data(DATA_IN_TYPE,DATA_USE_TYPE, complexScores)
        if GRIDSEARCH:
            bestScore, bestEst, scores = grid_search(featureData,complexScores,cutoff=30000)
            print(analyzeScores(scores))
            print(str(bestScore))
            print(bestEst)
        #print(test_of_a_test(featureData, complexScores))
        rawDat = five_fold_test(featureData, complexScores)
        #rawDat = please_work(featureData, complexScores)
        #rawDat = please_work_sk(featureData, complexScores)
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
        #rawDat = None
        precision = analyzer.calc_precision(processedData)
        recall = analyzer.calc_recall(processedData)
        print(getStateAsString())
        print('[simpleCorrect, simpleIncorrect, complexCorrect, complexIncorrect]:')
        print([len(category) for category in processedData])
        print('% categorically correct')
        print(a.calc_percent_categorically_right(processedData))
        report = classification_report(rawDat[1],rawDat[0])
        print(report)
        # sc = a.getScorer()
        # rawDat[0] = convert_data('bi_arr','bi_num',rawDat[0])
        # print(sc(rawDat[1],rawDat[0]))
