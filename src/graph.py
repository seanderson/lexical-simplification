"""
This module is written so as to represent the complex-simple relationships that
can be extracted from multiple sources in form of a graph. Then, this graph
can be used to predict the relative complexity of a given pair of words
"""


import re
import classpaths as paths
import newselautil as ns
from nltk.stem import WordNetLemmatizer;
Lemmatizer = WordNetLemmatizer()

SPPDB = paths.BASEDIR + "/SimplePPDB/simplification-dictionary"
# Source: http://www.seas.upenn.edu/~nlp/resources/simple-ppdb.tgz
SPPDB_used = paths.BASEDIR + "/SimplePPDB/simpple-simple-ppdb"
OED = paths.BASEDIR + "/OED.txt"  # Source:
# https://github.com/sujithps/Dictionary/blob/master/Oxford%20English%20Dictionary.txt
OED_used = paths.BASEDIR + "/Simple_OED.txt"


def test_sentence(sent_data, debug=False):
    fail = 0
    correct = 0
    incorrect = 0
    for i in range(len(sent_data)):
        for j in range(i + 1, len(sent_data)):
            diff = sent_data[i][1] - sent_data[j][1]
            if diff > 0:
                diff = 1
            elif diff < 0:
                diff = -1
            else:
                diff = 0
            node1 = Node.get_node(sent_data[i][0])
            node2 = Node.get_node(sent_data[j][0])
            if not node1 or not node2 or diff == 0:
                fail += 1
                continue
            score = Node.calculate_complexity(node1, node2)
            if not score:
                fail += 1
            elif score == diff:
                correct += 1
            else:
                incorrect += 1
                # if debug:
                    # print("\n\n " + node1.word + "\t" + node2.word + "\t" + str(
                        # diff))
                    # Node.DEBUG = True
                    # Node.calculate_complexity(node1, node2)
                    # Node.DEBUG = False
    return fail, correct, incorrect


def evaluate_sentence(sent_data):
    sent_data = [[x, 0, 0] for x in sent_data]
    for i in range(len(sent_data)):
        for j in range(i + 1, len(sent_data)):
            node1 = Node.get_node(sent_data[i][0])
            node2 = Node.get_node(sent_data[j][0])
            if not node1 or not node2:
                continue
            complex = Node.calculate_complexity(node1, node2)
            if complex:
                sent_data[i][2] += 1
                sent_data[i][2] += 1
                if complex == 1:
                    sent_data[i][1] += 1
                else:
                    sent_data[j][1] += 1
    for i in range(len(sent_data)):
        if sent_data[i][2] == 0:
            sent_data[i] = [sent_data[i][0], 0]
        else:
            sent_data[i] = [sent_data[i][0], float(sent_data[i][1])/sent_data[i][2]]
    return sent_data


def evaluate_lines(sentences_file, features_file):
    count = 0
    with open(sentences_file) as file:
        sents = file.readlines()
    with open(features_file) as file:
        features = file.readlines()
        for i in range(len(features)):
            features[i] = features[i].rstrip('\n').split('\t')
    s_i = 0
    f_i = 1
    while s_i < len(sents):
        sent_data = []
        if s_i % 2 == 0:
            print("Progress: " + str(round(float(s_i) / len(sents) * 100, 3)) + "%")
        word, w_i, _, sent, art, indx = sents[s_i].rstrip('\n').split('\t')
        previous = (sent, art, indx)
        while sent == previous[0]:
            if not re.match('.*[^a-zA-Z].*', word):
                word = Lemmatizer.lemmatize(word.lower())
                sent_data.append(word)
            s_i += 1
            if s_i == len(sents):
                break
            word, w_i, _, sent, art, indx = sents[s_i].rstrip('\n').split('\t')
        all_sent = [x for x in sent_data]
        for word in previous[0].split(' '):
            if re.match('.*[^a-zA-Z].*', word) or word in ns.STOPWORDS:
                continue
            word = Lemmatizer.lemmatize(word.lower())
            if word not in all_sent:
                sent_data.append(word)
        sent_data = evaluate_sentence(sent_data)
        j = 0
        while f_i < len(features) and features[f_i][0] == previous[1] and features[f_i][1] == previous[2]:
            match = features[f_i][-1].rstrip('\n').lower()
            if re.match('.*[^a-zA-Z].*', match):
                print("Not lemmatizable: " + match)
                features[f_i].insert(-1, '0')
            else:
                match = Lemmatizer.lemmatize(match)
                while sent_data[j][0] != match:
                    j += 1
                count += 1
                features[f_i].insert(-1, str(sent_data[j][1]))
            f_i += 1
    print("Count, length: " + str([count, len(features)]))
    features = '\n'.join(['\t'.join(x) for x in features])
    with open(features_file, 'w') as file:
        file.writelines(features)


def process_lines(lines, train_rather_than_test, debug=True):
    i = 0
    fail = 0
    correct = 0
    incorrect = 0
    while i < len(lines):
        if debug and not train_rather_than_test and i % 10 == 0:
            print("Progress: " + str(round(float(i)/len(lines) * 100, 3)) + "%")
        sent_data = []
        word, _, score, sent = lines[i].rstrip('\n').split('\t')
        score = int(score)
        previous = sent
        while sent == previous:
            if re.match('.*[^a-zA-Z].*', word):
                if not debug:
                    print("Word has non ASCII characters in it: " + word)
            else:
                word = Lemmatizer.lemmatize(word.lower())
                sent_data.append((word, score))
            i += 1
            if i == len(lines):
                break
            word, _, score, sent = lines[i].rstrip('\n').split('\t')
            score = int(score)
        all_sent = [x[0] for x in sent_data]
        for word in previous.split(' '):
            if re.match('.*[^a-zA-Z].*', word) or word in ns.STOPWORDS:
                continue
            word = Lemmatizer.lemmatize(word.lower())
            if word not in all_sent:
                sent_data.append((word, 0))
        if train_rather_than_test:
            Node.add_data(sent_data)
        else:
            f, c, ic = test_sentence(sent_data)
            fail += f
            correct += c
            incorrect += ic
    if not train_rather_than_test:
        Node.all_nodes = {}
        return fail, correct, incorrect


def create_simple_PPDB():
    """
    Extract useful information from the SimplePPDB database and write the data
    to an alone-standing file
    :return:
    """
    with open(SPPDB) as file:
        lines = file.readlines()
    with open(SPPDB_used, 'w') as file:
        for line in lines:
            if ' ' not in line:  # i.e. when the paraphrase is a word-to-word
                file.write(line)
                continue
            _, _, _, phrase0, phrase1 = line.lower().rstrip('\n').split('\t')
            w0 = [x for x in phrase0.split(' ') if x not in phrase1.split(' ')]
            w1 = [x for x in phrase1.split(' ') if x not in phrase0.split(' ')]
            if not (len(w0) == 1 and len(w1) == 1):
                # there is no one-to-one match.
                continue
            if re.match('.*[^a-z].*', w0[0]) or re.match('.*[^a-z].*', w1[0]):
                # there are some strangle symbols in one of these words
                continue
            if w1[0] in ns.STOPWORDS or w1[0] in ns.STOPWORDS:
                # stopwords should not be simplified
                continue
            file.write('\t'.join(line.split('\t')[:-2] + w0 + [w1[0] + "\n"]))


def create_simple_OED():
    """
    Extract useful information from the OED database and write the data
    to an alone-standing file
    :return:
     """
    with open(OED) as file:
        lines = file.readlines()
    with open(OED_used, 'w') as file:
        word = ""
        for line in lines:
            line = re.sub('(\[.*?\]|\(.*?\))', '', line)
            line = re.sub('(\(.*|\[.*)', '', line)
            # everything in parentheses and braces is information that can be
            # discarded (word etymology, etc.).
            line = re.sub('[,;:]', ' ', line)
            # this is so as to ensure that the word followed by a comma/colon
            # is still regarded as word (i.e.) that such words do not have
            # any non-letter characters in them
            line = re.sub(' [ ]*', ' ', line).lower().split(' ')
            if re.match('.*[^a-z].*', line[0]) or line[0] == word:
                # if line = word, then this is not a definition for a word
                # (which was on the previous line), but rather the definition
                # of a phrase containing that word
                continue
            word = line[0]
            definition = ""
            i = 1
            while i < len(line):
                if re.match('.*[^a-z].*', line[i]) or line[i] in ns.STOPWORDS:
                    continue
                definition = line[i]
                break
                i += 1
            if definition != "":
                definition = Lemmatizer.lemmatize(definition)
                file.write(word + "\t" + definition + "\n")


def load_simple_PPDB():
    """
    Load the preprocessed PPDB.
    :return:
    """
    with open(SPPDB_used) as file:
        lines = file.readlines()
    for line in lines:
        score1, score2, _, complex, simple = line.split('\t')
        simple = simple.rstrip('\n')
        # Score1 varies from 1 to 5 (validity score from the original PPDB)
        # Score2 varies from 0.5 to 1 (simplification score)
        # Hence, float(score1) * float(score2) * 2 - 1varies from 0 to 9, just
        # like teh score in the Kriz paper. However, a different metric can
        # be used
        score = float(score1) * float(score2) * 2 - 1
        Node.add_data([(complex, score), (simple, 0)])


def load_simple_OED():
    """
    Load the preprocessed OED
    :return:
    """
    with open(OED_used) as file:
        lines = file.readlines()
    for line in lines:
        complex, simple = line.split('\t')
        simple = simple.rstrip('\n')
        # The data in OED is messy which is why it is always assigned teh weight
        # lower than that from the Kriz paper or SPPDB
        Node.add_data([(complex, 1), (simple, 0)])


class Node:
    """
    A class that represents a node in a graph
    """

    DEBUG = False
    all_nodes = {}  # a dictionary of all the nodes ever created

    def __init__(self, word):
        """
        This method should only be used inside get_node() function
        :param word:
        """
        if word in Node.all_nodes:
            print("WW: the node '" + word + "'already exists")
        Node.all_nodes[word] = self
        self.word = word
        self.edges = {}
        # dictionary containing information about the edges from this node

    def add(self, node, diff):
        """
        Add an edge from node "node" with value "diff"
        :param node:
        :param diff:
        :return:
        """
        if node not in self.edges:
            self.edges[node] = []
        # at thi stage, there might be multiple edges from one node to another
        self.edges[node].append(diff)

    @staticmethod
    def contract_data():
        """
        A method that merges all the edges between any pair of edges into a
        single edge by summing all the weights on these edges. Adding the
        weights seems to work better than averaging them, because multiple
        connections increase the confidence of the connection
        :return:
        """
        for name in Node.all_nodes.keys():
            node = Node.get_node(name)
            # length = len(node.edges.keys())  # uncomment if averaging is used
            for key in node.edges.keys():
                node.edges[key] = float(sum(node.edges[key]))
                #  node.edges[key] /= length  # uncomment if averaging is used

    def compare_to(self, cmp, depth=2, scale=1):
        """
        Compares the complexity of self to that of cmp
        :param cmp:     The node to which to compare self
        :param depth:   The algorithm will search for all paths from self to cmp
        such that there are no more that "depth" nodes inbetween
        :param scale:   Numbers x and x / scale > y > x * scale would be
                        considered equal (this feature is currently not used)
        :return:        1, if self > cmp
                        -1, if self < cmp
                        0, if self == cmp
        """
        if Node.DEBUG:
            print("Paths from " + self.word + " to " + cmp.word + ":")
        a = self.reach(cmp, depth)
        if Node.DEBUG:
            print("Paths from " + self.word + " to " + cmp.word + ":")
        b = cmp.reach(self, depth)
        if a == b and b == -1:  # -1 means there is no path between the nodes
            return None
        if a > scale * b:
            return 1
        if b > scale * a:
            return -1
        else:
            return 0

    def reach(self, node, depth=2):
        """
        Find all paths from self to node such that they pass through no more
        than deep intermediate nodes
        :param node:
        :param depth:
        :return: the sum of all different paths (where the value of a path is
        the sum of teh weights on the edges along the way)
        """
        result = []
        if self == node:
            result = [0]
            if Node.DEBUG:
                print("WW: comparing a node to itself is a waste of time")
        elif depth == 0:
            if node in self.edges:
                if Node.DEBUG:
                    print(self.word + "->" + node.word + " (" + str(round(
                        self.edges[node], 2)) + ")")
                result = [self.edges[node]]
        else:
            for key in self.edges.keys():
                curr = key.reach(node, depth - 1)
                if curr != -1:
                    if Node.DEBUG:
                        print(self.word + "->" + key.word + " (" + str(round(
                            self.edges[key], 2)) + ")")
                    result.append(self.edges[key] + curr)
        if len(result) == 0:
            return -1
        # return float(sum(result)) / len(result)  # uncomment to use averaging
        # insetad of summation
        return float(sum(result))

    @staticmethod
    def get_node(word, create_new_node=False):
        """
        Get node from the word. If create_new_node = True and a node
        representing the word does not exist, create such node
        :param create_new_node:
        :return: The node whose word field equals the word parameter of this
        method
        """
        if not re.match('[a-zA-Z]*', word):
            print("WW: Node " + word + " should not be created")
        if word not in Node.all_nodes:
            if create_new_node:
                Node.all_nodes[word] = Node(word)
            else:
                return None
        return Node.all_nodes[word]

    @staticmethod
    def add_data(data):
        """
        Add different connections to the graph
        :param data: A list of tuples, where each tuple consists of two
        elements. The first is a word (string corresponding to a node) and the
        second is the complexity score assigned to it. The difference between
        the complexity scores of two words will be the weight of an edge
        :return:
        """
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                diff = data[i][1] - data[j][1]
                if diff > 0:
                    Node.get_node(data[i][0], True).add(
                        Node.get_node(data[j][0], True), diff)
                elif diff < 0:
                    Node.get_node(data[j][0], True).add(
                        Node.get_node(data[i][0], True), diff)


def main(fraction):
    # print('Creating PPDB...')
    # create_simple_ppdb()
    with open(paths.KRIZ_PAPER_FILE) as file:
        lines = file.readlines()
    fr_size = int(fraction * len(lines))
    for i in range(int(1/fraction)):
        train = lines[0: i * fr_size] + lines[(i + 1) * fr_size:]
        test = lines[i * fr_size: (i + 1) * fr_size]
        print('\nLoading PPDB...')
        load_simple_PPDB()
        print('Loading OED...')
        load_simple_OED()
        print('Loading newsela data...')
        process_lines(train, True)
        Node.contract_data()
        print('Testing...')
        fail, correct, incorr = process_lines(test, False)
        print('Run #' + str(i) + ': f, c, i = ' + str([fail, correct, incorr]))


if __name__ == "__main__":
    create_simple_PPDB()
    # main(0.2)
    # create_simple_OED()
    # print('\nLoading PPDB...')
    # load_simple_ppdb()
    # print('Loading OED...')
    # load_simple_OED()
    # Node.contract_data()
    # print('Testing...')
    # evaluate_lines(paths.NEWSELA_COMPLEX + "/Newsela_Complex_Words_Dataset_supplied.txt",
    #               paths.NEWSELA_COMPLEX + "/testFeatClass.txt")
    """ with open(paths.NEWSELA_COMPLEX + "/testFeatClass_copy.txt") as file:
        lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].split("\t")[-2] + "\t" + lines[i].split('\t')[-1]
    with open(paths.NEWSELA_COMPLEX + "/testFeatClass.txt", "w") as file:
        file.writelines(lines) """



