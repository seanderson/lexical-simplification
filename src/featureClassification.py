"""
Classifies words as complex or simple using methods based on Reno Kriz's
Simplification Using Paraphrases and Context-based Lexical Substitution
"""
import re
import numpy
from lexenstein.identifiers import *
from lexenstein.features import *
from lexenstein.morphadorner import MorphAdornerToolkit
import classpaths as paths
from nltk.corpus import wordnet
from nltk.corpus import cmudict
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import  MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import copy
import random


CWICTORIFY = False
TESTCLASSIFY = False
IMPORTDATA = False
NNET = True
GRIDSEARCH = False
BINARY_CATEGORIZATION = True
BINARY_EVALUATION = BINARY_CATEGORIZATION or True
ALL_COMPLEX = False
REMOVE_ZEROS = False
DEBUG = False

WORD_ONLY_CONFIG = [True, True, True, True, True, False, False, False, False, False, False, False]
CONTEXT_ONLY_CONFIG = [False, False, False, False, False, True, True, True, True, True, True, True]
ALL_FEATURES_CONFIG = [True, True, True, True, True, True, True, True, True, True, True, True]
USE_WORD_VECS = True
CURRENT_CONFIG = ALL_FEATURES_CONFIG

NEWSELLA_SUPPLIED = paths.NEWSELA_COMPLEX + "Newsela_Complex_Words_Dataset_supplied.txt"
CWICTOIFIED = paths.NEWSELA_COMPLEX + "Cwictorified.txt"
SAVE_FILE = paths.NEWSELA_COMPLEX + "Feature_data.txt"
GRAPH_FILE = paths.NEWSELA_COMPLEX + "Graph_output.txt"
VEC_FILE = paths.NEWSELA_COMPLEX + "word_embeddings_Jul-05-1256_epoch0.tsv"


def getStateAsString():
    s = ''
    s += 'NNET = ' + str(NNET) + '\n'
    s += 'BINARY_CATEGORIZATION = ' + str(BINARY_CATEGORIZATION) + '\n'
    s += 'ALL_COMPLEX = ' + str(ALL_COMPLEX) + '\n'
    s += 'REMOVE_ZEROS = ' + str(REMOVE_ZEROS) + '\n'
    s += 'DEBUG = ' + str(DEBUG) + '\n'
    return s


def cwictorify(inputPath, outputPath):
    """
    Writes the file from inputPath in CWICTOR format
    :param inputPath:
    :param outputPath:
    """
    # format: Sentence   word    indexInSent     BinaryIsComplex
    with open(inputPath) as file:
        input = file.readlines()
    with open(outputPath,"w") as output:
        for line in input:
            list = line.split('\t')
            #print(list)
            if int(list[2]) > 3:
                c = 1
            else:
                c = 0
            output.write(list[3]+"\t"+list[0]+"\t"+list[1]+"\t"+str(c)+"\n")
    return outputPath


def save(data, outPath):
    """
    Saves data to a file at outPath
    :param data:
    :param outPath:
    """
    l =[]
    with open(outPath, 'w') as out:
        out.write('ArticleName SentInd(Article) WordInd(Sentence) WordLength ' +
                  'NumSynonyms NumSynsets WordSyllables 1GramFreq GraphScore AvgSentSylbs SentLength AvgWordLen AvgNumSynonyms AvgNumSynsets Avg1GramFreq\n')
        for line in data:
            s = ''
            for i in range(len(line)-1):
                s += str(line[i]) + '\t'
            s += str(line[len(line)-1])
            l.append(s+'\n')
        out.writelines(l)
    print("Data Saved")


def count_sentence_syllables(sent, d = cmudict.dict(), m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)):
    """
    counts the number of syllables in words (strings separated by spaces that
     contain letters) in a  given sentence
    :param sent: the sentence as a string, punctuation separated by spaces
    :return: the number of syllables
    """
    words = sent.split(' ')
    syllables = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            try:
                syllables += count_word_syllables(word, d, m)[0]
            except:
                syllables += count_word_syllables(word, d, m)
        else:
            words.remove(word)
    return float(syllables)/float(len(words))


def count_word_syllables(word, d = cmudict.dict(), m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)):
    """
    Counts the syllables in a word
    :param word: the word to be counted
    :return: the number of syllables
    """
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        return len(m.splitSyllables(word)[0].split('-'))


def calc_sent_len(sent):
    """
    Calculates the number of words in a sentence
    :param sent: the sentence as a string
    :return: the number of words in sent
    """
    words = sent.split(' ')
    length = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            length += 1
    return length


def calc_avg_word_lens(sent):
    """
    Calculates the average length of the words in a sentence
    :param sent: the sentence as a string
    :return: the average number of letters in the words in sent
    """
    words = sent.split(' ')
    totalLen = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            totalLen += len(word)
        else:
            words.remove(word)
    return float(totalLen) / float(len(words))


def calc_syn_avgs(sent):
    """
    Calculates the average number of synonyms of the words in a sentence. Will
    skip words that contain characters unreadable by wordnet
    :param sent: the sentence as a string
    :return: the  average number of synonyms of the words in sent
    """
    words = sent.split(' ')
    totalSyns = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            if not re.match(r'.*[^ -~].*', word):
                for syn in wordnet.synsets(word):
                    totalSyns += len(syn.lemmas())
            else:
                print(word)
        else:
            words.remove(word)
    return float(totalSyns) / float(len(words))


def calc_synset_avgs(sent):
    """
    Calculates the average number of synsets of the words in a sentence. Will
    skip words that contain characters unreadable by wordnet
    :param sent: the sentence as a string
    :return: the  average number of synsets of the words in sent
    """
    words = sent.split(' ')
    totalSets = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            if not re.match('.*[^ -~].*', word):
                totalSets += len(wordnet.synsets(word))
            else:
                print(word)
        else:
            words.remove(word)
    return float(totalSets) / float(len(words))


def calc_nGram_avgs(sent, ngramDict, size):
    """
    calculates the average google 1-gram frequencies of the words in a sentence
    :param sent: the sentence as a string
    :param ngramDict: a dictionary of {word, number of appearances in google 
        1-gram}
    :param size: the total number of the appearances of all words in the 1-gram
    :return: he average google 1-gram frequencies of the words in sent
    """
    words = sent.split(' ')
    totalAvg = 0
    for word in words:
        if re.match('.*[a-zA-Z].*', word):
            if word in ngramDict:
                totalAvg += float(ngramDict[word]) / size
        else:
            words.remove(word)
    return float(totalAvg) / float(len(words))


def collect_data(corpusPath, CWPath, vecPath):
    """
    Collects features from a corpus in CWICTOR format from a file at CWPath
    and a file in Kriz format at corpusPath
    :param corpusPath:
    :param CWPath:
    :return: the list of features
    """
    d = cmudict.dict()
    m = MorphAdornerToolkit(paths.MORPH_ADORNER_TOOLKIT)

    fe = FeatureEstimator()
    fe.addLengthFeature('Complexity')  # word length
    fe.addSynonymCountFeature('Simplicity')  # WordNet synonyms
    list = fe.calculateFeatures(cwictorify(corpusPath, CWPath), format='cwictor')

    sentenceSylbs = []
    currentArticle = ""

    with open(CWPath) as out:
        lines = out.readlines()
    with open(corpusPath) as corp:
        orig = corp.readlines()
    if USE_WORD_VECS:
        with open(vecPath) as vec:
            vecs = vec.readlines()

    if(DEBUG):
        lines = lines[:100]
        orig = orig[:100]
        list = list[:100]

    sOrig = [j.split('\t') for j in orig]

    # prep 1-gram dictionary
    with open(paths.USERDIR + "/data/web1T/1gms/vocab") as file:
        ngrams = file.readlines()
    for lineNum in range(len(ngrams)):
        ngrams[lineNum] = ngrams[lineNum].split('\t')
    ngramDict = {x[0]: int(x[1]) for x in ngrams}
    size = int(open(paths.USERDIR + "/data/web1T/1gms/total").read())

    # prep graph file
    graphScores = []
    with open(GRAPH_FILE) as file:
        tmp = file.readlines()
        tmp = tmp [1:]
    for lineNum in range(len(tmp)):
        tmp[lineNum] = tmp[lineNum].split('\t')
        graphScores.append(tmp[lineNum][0])

    print("files read")

    # append lines
    for i in range(len(list)):
        #print(i)
        line = lines[i].split('\t')
        
        # unique WordNet synsets
        if not re.match(r'.*[^ -~].*', line[1]):
            list[i].append(len(wordnet.synsets(line[1])))
        else:
            list[i].append(0)
        # number of syllables
        list[i].append(count_word_syllables(line[1], d, m))
        # google 1-gram freq
        if line[1] in ngramDict:
            #list[i].append(float(ngramDict[line[1]]) / size)
            list[i].append(ngramDict[line[1]])
        else:
            list[i].append(0)

        # graph score
        list[i].append(graphScores[i])

        #reset sentence features
        index = int(sOrig[i][-1])
        if currentArticle != sOrig[i][-2]:
            currentArticle = sOrig[i][-2]
            sentenceSylbs = []
            sentLens = []
            wordLenAvgs = []
            synonymCountAvgs = []
            synsetNumAvgs = []
            nGramFreqAvgs = []
        # update sentence features
        while len(sentenceSylbs) < index+1:
            sentenceSylbs.append(count_sentence_syllables(sOrig[i][3], d, m))
            sentLens.append(calc_sent_len(sOrig[i][3]))
            wordLenAvgs.append(calc_avg_word_lens(sOrig[i][3]))
            synonymCountAvgs.append(calc_syn_avgs(sOrig[i][3]))
            synsetNumAvgs.append(calc_synset_avgs(sOrig[i][3]))
            nGramFreqAvgs.append(calc_nGram_avgs(sOrig[i][3], ngramDict, size))

        # number of sentence syllables
        list[i].append(sentenceSylbs[index])
        # sent length
        list[i].append(sentLens[index])
        # avg length of words in sentence
        list[i].append(wordLenAvgs[index])
        # avg synonym count in sentence
        list[i].append(synonymCountAvgs[index])
        # avg num synsets in sentence
        list[i].append(synsetNumAvgs[index])
        # avg word 1-gram freq in sentence
        list[i].append(nGramFreqAvgs[index])

        if USE_WORD_VECS:
            vecvals = vecs[i].split('\t')[1:]
            for val in vecvals:
                list[i].append(val)

        list[i].insert(0, line[2])
        list[i].insert(0, sOrig[i][-1].strip('\n'))
        list[i].insert(0, sOrig[i][-2])
        list[i].append(sOrig[i][0])    # TODO make 10x10 confusion matrix if NNet
        # list.append(line[1])   #causes file to be unreadable?
        if i % 50 == 0:
            print(str(i) + " out of " + str(len(list)))
    return list


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
        data.append(line.split('\t')[3:-1])
    data = data[1:]
    if featureConfig == -1:
        return data
    for featureSetInd in range(len(data)):
        featureSet = data[featureSetInd]
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
    return data


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
    return complexities


def classify(data):
    """
    trains a SVM on data
    :param data: the data to train the SVM on. In format [X,Y]
    :return: the trained SVM
    """
    clf = svm.SVC(C=1000.0, cache_size=500, gamma=1, kernel='rbf')
    #clf = svm.SVC(kernel='rbf', C=1, verbose=False, probability=False, degree=3, shrinking=True, max_iter = -1, decision_function_shape='ovr', random_state=None, tol=0.001, cache_size=200, coef0=0.0, gamma=0.1, class_weight=None)
    clf.fit(data[0], data[1])
    return clf


def str_to_bin_category(item):
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
    available = [tempX, tempY]
    if not BINARY_CATEGORIZATION:
        available[1] = map(int, available[1])
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
        # SVM
        #clf = LogisticRegression()
        #clf.fit(train[0], train[1])
        if NNET:
            clf = MLPClassifier()
            clf.fit(train[0], train[1])
        else:
            clf = classify(train)
        preds = clf.predict(test[0])
        # results.append(calc_percent_right(test, preds))
        results[0] = np.append(results[0], preds)
        results[1] = np.append(results[1], test[1])
    return results


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


def temp_kfold_test(X,Y):
    """
    tests scikit-learn's n-fold testing
    :param X:
    :param Y:
    :return:
    """
    clf = svm.SVC(cache_size= 500, kernel='rbf')
    X = preprocessing.scale(X)
    scores = cross_val_score(clf, X, Y, cv=5)
    return scores


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


def calc_TP(pData):
    TP = 0
    for i in range(len(pData[0])):
        TP += pData[i][i]
    return TP


def calc_FP(pData):
    FP = 0
    return FP


def calc_FN(pData):
    FN = 0
    return FN


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
    else:
        TP = calc_TP(pData)
        FP = calc_FP(pData)
    if TP + FP == 0:
        return 0
    return float(TP)/float(TP+FP)


def calc_recall(pData):
    if BINARY_EVALUATION:
        TP = len(pData[2])
        FN = len(pData[3])
    else:
        TP = calc_TP(pData)
        FN = calc_FN(pData)
    if TP + FN == 0:
        return 0
    return float(TP)/float(TP+FN)


def calc_f_measure(precision, recall):
    if precision + recall == 0:
        return -1
    return 2*precision*recall/(precision + recall)


def custom_f1_scorer(y, y_pred, **kwargs):
    if BINARY_CATEGORIZATION:
        data = process_results_bin([y_pred,y])
    else:
        data = process_results([y_pred,y])
    precision = calc_recall(data)
    recall = calc_recall(data)
    return calc_f_measure(precision, recall)


def grid_search(X, Y):
    print('doing grid search')
    '''if(BINARY_CATEGORIZATION):
        for i in range(len(Y)):
            Y[i] = str_to_bin_category(Y[i])'''
    if NNET:
        hiddenLayerSizes = [(100,),(90,),(80,),(70,),(60,),(50,),(40,),(30,),(20,),(10,),(1,)]
        activations = ['identity', 'logistic', 'tanh', 'relu']
        solvers = ['lbfgs', 'sgd', 'adam']
        learningRates = ['constant', 'invscaling', 'adaptive']
        alphas = 10.0**-numpy.arange(1,7)
        parameters = {'hidden_layer_sizes': hiddenLayerSizes, 'activation': ['tanh'], 'solver': ['adam'], 'learning_rate': ['adaptive'], 'alpha': alphas}
        if DEBUG:
            parameters = {}
        evaluator = MLPClassifier()
    else:
        #parameters = {'kernel': ['rbf'], 'C': [.01, .1, 1, 10, 100, 1000],
        #              'gamma': [.001,.01,.1,1,10,100,1000]}
        parameters = {'kernel': ['rbf'], 'C': [800, 900, 1000, 1100, 1200],
            'gamma': [.01, .5, 1, 5, 10]}
        if(DEBUG):
            parameters = {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [1, 10], 'early_stopping': [True]}
        evaluator = svm.SVC()
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    scorer = make_scorer(custom_f1_scorer, labels=['c'], average=None)
    clf = GridSearchCV(evaluator, parameters, scoring=scorer, verbose=3, n_jobs=8, cv=10)
    clf.fit(X,Y)
    scores = [x[1] for x in clf.grid_scores_]
    return clf.best_score_, clf.best_estimator_.get_params(), scores


if __name__ == '__main__':
    if(TESTCLASSIFY):
        iris = datasets.load_iris()
        rawDat = five_fold_test(iris.data, iris.target)
        processedData = []
        for i in range(len(rawDat[0])):
            processedData.append([rawDat[0][i],rawDat[1][i]])
        print(calc_percent_right(processedData))
    if(CWICTORIFY):
        cwictorify(NEWSELLA_SUPPLIED, CWICTOIFIED)
    if(IMPORTDATA):
        data = (collect_data(NEWSELLA_SUPPLIED, CWICTOIFIED, VEC_FILE))
        save(data, SAVE_FILE)
        #print (data)
    if not TESTCLASSIFY:
        config = CURRENT_CONFIG
        featureData = read_features(SAVE_FILE, config)
        complexScores = read_complexities(NEWSELLA_SUPPLIED)
        if REMOVE_ZEROS:
            tempX = []
            tempY = []
            for labelInd in range(len(complexScores)):
                if not (complexScores[labelInd] == 0 or complexScores[labelInd] == '0'):
                    tempX.append(complexScores[labelInd])
                    tempY.append(featureData[labelInd])
            featureData = tempY
            complexScores = tempX
        if BINARY_CATEGORIZATION:
            for labelInd in range(len(complexScores)):
                complexScores[labelInd] = str_to_bin_category(complexScores[labelInd])
        if GRIDSEARCH:
            bestScore, bestEst, scores = grid_search(featureData,complexScores)
            print(scores)
            print(str(bestScore))
            print(bestEst)
        rawDat = five_fold_test(featureData, complexScores)
        if ALL_COMPLEX:
            if BINARY_EVALUATION:
                for i in range(len(rawDat[0])):
                    rawDat[0][i] = 'c'
            else:
                for i in range((len(rawDat[0]))):
                    rawDat[0][i] = 9
        if BINARY_CATEGORIZATION:
            processedData = process_results_bin(rawDat)
        else:
            processedData = process_results(rawDat)
        precision = calc_precision(processedData)
        recall = calc_recall(processedData)
        print(getStateAsString())
        print([len(category) for category in processedData])
        print(calc_percent_categorically_right(processedData))
        print(precision, recall, calc_f_measure(precision, recall))
