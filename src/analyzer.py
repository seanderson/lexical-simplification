from sklearn.metrics import confusion_matrix
import featureClassification


class Analyzer:
    # num = int 0-9 (>3 = complex)
    # bi_num = 1 or 0
    # bi_str = 's' or 'c'
    # bi_arr = [1,0] (complex) or [0,1] (simple)

    DATA_TYPES = ['num', 'bi_num', 'bi_str', 'bi_arr']
    BINARY_TYPES = ['bi_num', 'bi_str', 'bi_arr']
    ARR_POSSIBILITIES = [[1, 0], [0, 1]]
    NUM_POSSIBILITIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    BI_NUM_POSSIBILITIES = [1, 0]
    STR_POSSIBILITIES = ['c', 's']

    def __init__(self, data_type):
        """
        initializes self.DATA_TYPE given an input in DATA_TYPES or an example of
        the data format
        :param data_type:
        """
        if data_type in Analyzer.DATA_TYPES:
            self.DATA_TYPE = data_type
            return
        elif data_type in self.ARR_POSSIBILITIES:
            self.DATA_TYPE = 'bi_arr'
            return
        elif data_type in self.NUM_POSSIBILITIES:
            self.DATA_TYPE = 'num'
            return
        elif data_type in self.STR_POSSIBILITIES:
            self.DATA_TYPES = 'bi_str'
            return
        elif data_type in self.BI_NUM_POSSIBILITIES:
            self.DATA_TYPES = 'bi_num'
            return
        print('ERROR: unrecognized data_type')

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
        :return: % correct
        """
        if len(processedDataCategory) == 0:
            return 0
        check = []
        for j in range(len(processedDataCategory)):
            check.append(
                processedDataCategory[j][0] == processedDataCategory[j][1])
        numRight = 0
        for i in check:
            if i:
                numRight += 1
        return float(numRight) / float(len(check))

    def process_results(self, results):
        """
        processes results for any format of data in DATA_TYPES
        :param results: [predicted categorizations, actual categorizations]
        :return: a confusion matrix
        """
        if self.DATA_TYPE in Analyzer.BINARY_TYPES:
            results = featureClassification.convert_data(self.DATA_TYPE,
                                                         'bi_str', results)
            return process_results_bi_str(results)
        else:
            results = featureClassification.convert_data(self.DATA_TYPES, 'num',
                                                         results)
            return process_results_reg(results)

    def calc_avg_percent_right(self, pData):
        """
        calculates the average percent right over multiple lists in pData
        :param pData: data that has been fed through one of the process_results
        methods
        :return: avg % right
        """
        avg = 0
        for i in range(len(pData)):
            avg += self.calc_percent_right(pData[i])
        avg /= i
        return avg

    def calc_percent_categorically_right(self, pData):
        """
        calculates the % right
        :param pData: data that has been fed through one of the process_results
        methods
        :return:
        """
        if self.BIN_EVAL:
            return float(len(pData[0]) + len(pData[2])) / \
                   float(sum([len(pData[0]), len(pData[1]), len(pData[2]),
                              len(pData[3])]))
        else:
            return 0

    def getScorer(self):
        """
        gets a scorer that fits DATA_TYPE
        :return: a scorer function
        """
        scorers = [reg_scorer, bi_num_scorer,
                   bi_str_scorer, bi_arr_scorer]
        for i in range(len(Analyzer.DATA_TYPES)):
            if Analyzer.DATA_TYPES[i] == self.DATA_TYPE:
                return scorers[i]
        print('ERROR: scorer not found')
        return None


def process_results_reg(results):
    '''
    reformats results into a confusion matrix
    :param results: [predicted categorizations, actual categorizations]
    :return: confusion matrix split into one list
    '''
    pred = []
    actual = []
    for i in range(len(results[0])):
        actual.append(int(results[1][i]))
        pred.append(int(results[0][i]))
    data = confusion_matrix(actual, pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return data


'''          correct    incorrect       A v P > complex  simple  
    complex     TP          FN          complex  CC TP  CI FN
     simple     TN          FP           simple  SI FP  SC TN
    [TN, FP, TP, FN]
'''


def process_results_bi_str(results):
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
                simpleCorrect.append([pred, right])
            else:
                simpleIncorrect.append([pred, right])
        else:
            if pred == 'c':
                complexCorrect.append([pred, right])
            else:
                complexIncorrect.append([pred, right])
    data = [simpleCorrect, simpleIncorrect, complexCorrect,
            complexIncorrect]
    return data


def calc_TP(pData):
    TP = 0
    for i in range(len(pData[0])):
        TP += pData[i][i]
    return TP


def calc_precision(pData):
    """
    :param pData: data that has been fed through one of the process_results
    methods
    :return: precision
    """
    TP = len(pData[2])
    FP = len(pData[1])
    if TP + FP == 0:
        return 0
    return float(TP) / float(TP + FP)


def calc_recall(pData):
    """
    :param pData: data that has been fed through one of the process_results
    methods
    :return: recall
    """
    TP = len(pData[2])
    FN = len(pData[3])
    if TP + FN == 0:
        return 0
    return float(TP) / float(TP + FN)


def calc_f_measure(precision, recall):
    """
    calculates f score from precision and recall
    :return: f measure
    """
    if precision + recall == 0:
        return -1
    return 2 * precision * recall / (precision + recall)


def reg_scorer(y, y_pred, **kwargs):
    """
    scores data if data is type num
    :param y: actual labels
    :param y_pred: predicted labels
    :param kwargs:
    :return: the f score
    """
    data = process_results_reg([y_pred, y])
    precision = calc_recall(data)
    recall = calc_recall(data)
    return calc_f_measure(precision, recall)


def bi_str_scorer(y, y_pred, **kwargs):
    """
    scores data if data is type bi_str
    :param y: actual labels
    :param y_pred: predicted labels
    :param kwargs:
    :return: the f score
    """
    data = process_results_bi_str([y_pred, y])
    precision = calc_recall(data)
    recall = calc_recall(data)
    return calc_f_measure(precision, recall)


def bi_num_scorer(y, y_pred, **kwargs):
    """
    scores data if data is type bi_num
    :param y: actual labels
    :param y_pred: predicted labels
    :param kwargs:
    :return: the f score
    """
    y = featureClassification.convert_data('bi_num', 'bi_str', y)
    y_pred = featureClassification.convert_data('bi_num', 'bi_str', y_pred)
    data = process_results_bi_str([y_pred, y])
    precision = calc_recall(data)
    recall = calc_recall(data)
    return calc_f_measure(precision, recall)


def bi_arr_scorer(y, y_pred, **kwargs):
    """
    scores data if data is type bi_arr
    :param y: actual labels
    :param y_pred: predicted labels
    :param kwargs:
    :return: the f score
    """
    y = featureClassification.convert_data('bi_arr', 'bi_str', y)
    y_pred = featureClassification.convert_data('bi_num', 'bi_str', y_pred)
    data = process_results_bi_str([y_pred, y])
    precision = calc_recall(data)
    recall = calc_recall(data)
    return calc_f_measure(precision, recall)


def custom_f1_scorer(y, y_pred, **kwargs):
    """
    scores data if data is in DATA_TYPES
    :param y: actual labels
    :param y_pred: predicted labels
    :param kwargs:
    :return: the f score
    """
    a = Analyzer(y[0])
    scorer = a.getScorer()
    return scorer(y, y_pred, **kwargs)
