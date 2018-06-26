"""
This module is written so as to represent the complex-simple relationships that
can be extracted from multiple sources in form of a graph. Then, this graph
can be used to predict the relative complexity of a given pair of words
"""


import re
import classpaths as paths
import newselautil as ns
from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()

SPPDB = paths.BASEDIR + "/SimplePPDB/simplification-dictionary"
# Source: http://www.seas.upenn.edu/~nlp/resources/simple-ppdb.tgz
SPPDB_used = paths.BASEDIR + "/SimplePPDB/simpple-simple-ppdb"
OED = paths.BASEDIR + "/OED.txt"  # Source:
# https://github.com/sujithps/Dictionary/blob/master/Oxford%20English%20Dictionary.txt
OED_used = paths.BASEDIR + "/Simple_OED.txt"


def test_sentence(data, limit, debug=False):
    """
    Test the performance of the graph on a sentence from the Kriz paper data.
    :param data:  A list of tuples, where each tuple consists of two
    elements. The first is a word (string corresponding to a node) and the
    second is the complexity score assigned to it in the Kriz data.
    :param limit: The length of that part of data for which the Kriz paper
    has explicitly results
    :param debug:
    :return:
    """
    no_diff = 0    # number of times the Kriz paper assigns equal scores
    fail = 0       # number of times (out of all - no_diff) the graph fails
    #                to make a prediction
    correct = 0    # number of times the graph makes a correct prediction
    # correct means that the graph identifies what word is more complex
    incorrect = 0  # number of times the graph makes an incorrect prediction
    # no_diff + fail + correct + incorrect = sum(i) for i = 1 to len(data) - 1
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            diff = data[i][1] - data[j][1]
            if diff != 0:
                diff = diff/abs(diff)  # 1 or -1
            else:
                no_diff += 1
                continue
            score = Node.compare_two_words(data[i][0], data[j][0])
            if not score:
                fail += 1
            elif score == diff:
                correct += 1
            else:
                incorrect += 1
                if debug:
                    print("\n\n " + data[i][0] + "\t" + data[j][0]
                          + "\t" + str(diff))
                    Node.DEBUG = True
                    Node.compare_two_words(data[i][0], data[j][0])
                    Node.DEBUG = False
    return no_diff, fail, correct, incorrect


def evaluate_sentence(data, limit):
    """
    Get a metric of how complex the word is relative to its context by
    calculating the number of words that are simpler than it and dividing this
    value by the number of words it could be compared to
    :param data: A list of words
    :param limit: The length of that part of data for which the Kriz paper
    has explicitly results
    :return:
    """
    data = [[x[0], 0, 0] for x in data[:limit]]
    # data[i][2] is the number of words that i-th word was compared to
    # data[i][1] is the number of times the i-th word was more complex out of
    # the data[i][2] comparisons
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            score = Node.compare_two_words(data[i][0], data[j][0])
            if score and score != 0:
                data[i][2] += 1
                data[i][2] += 1
                if score == 1:
                    data[i][1] += 1
                else:
                    data[j][1] += 1
    for i in range(len(data)):
        if data[i][2] == 0:
            data[i] = [data[i][0], 0]
        else:
            data[i] = [data[i][0], float(data[i][1])/data[i][2]]
            # TODO: Perhaps, data[i][1] = pow(data[i][1], 2) would make sense
    return data


def process_lines(lines, function, debug=False, print_progress=50):
    """
    Iterate over the lines from teh Kriz paper and apply a function to them
    :param function:
    :param lines:
    :param debug:
    :param print_progress: if debug, print progress each print_progress lines
    :return:
    """
    i = 0
    result = []
    while i < len(lines):
        if debug and i % 10 == 0:
            print("Progress: " + str(round(float(i)/len(lines) * 100, 3)) + "%")
        sent_data = []
        word, _, score, sent = lines[i].rstrip('\n').split('\t')
        score = int(score)
        previous = sent
        while sent == previous:
            word = word.lower()
            if not re.match('.*[^a-zA-Z].*', word):
                word = Lemmatizer.lemmatize(word)
            sent_data.append((word, score))
            i += 1
            if i == len(lines):
                break
            word, _, score, sent = lines[i].rstrip('\n').split('\t')
            score = int(score)
        marked_words = [x[0] for x in sent_data]
        # words for which the complexity is explicitly stated
        for word in previous.split(' '):
            if re.match('.*[^a-zA-Z].*', word) or word in ns.STOPWORDS:
                continue
            word = Lemmatizer.lemmatize(word.lower())
            if word not in marked_words:
                sent_data.append((word, 0))
        result.append(function(sent_data, len(marked_words)))
    return result


def test(lines):
    """
    Test the algorithm on the lines from the Kriz paper. The meaning of the
    result is explained in teh docstring for test_sentence
    :param lines:
    :return:
    """
    result = process_lines(lines, test_sentence)
    no_diff = sum([x[0] for x in result])
    fail = sum([x[1] for x in result])
    correct = sum([x[2] for x in result])
    incorrect = sum([x[3] for x in result])
    return no_diff, fail, correct, incorrect


def evaluate(lines):
    """
    Use the algorithm to assess the complexity of words in Kriz data
    :param lines:
    :return:
    """
    result = process_lines(lines, evaluate_sentence())
    output = []
    for sent in result:
        for word in sent:
            output.append(str(word[1]))
    return output


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
    def add_data(data, _=None):
        """
        Add different connections to the graph
        :param data: A list of tuples, where each tuple consists of two
        elements. The first is a word (string corresponding to a node) and the
        second is the complexity score assigned to it. The difference between
        the complexity scores of two words will be the weight of an edge
        :param _: a dummy parameter.
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

    @staticmethod
    def compare_two_words(word1, word2):
        node1 = Node.get_node(word1)
        node2 = Node.get_node(word2)
        if not node1 or not node2:
            return None
        return node1.compare_to(node2)


def main(fraction, test_data=True, output_file=None):
    """
    Process the Kriz data
    :param fraction: if fraction != 1, do a 1/fraction - fold testing
    :param test_data: if True, test the performance right away.
                      if False, output the evaluations to a file
    :param output_file:
    :return:
    """
    with open(paths.KRIZ_PAPER_FILE) as file:
        lines = file.readlines()
    fr_size = int(fraction * len(lines))
    if not test_data and output_file is None:
        print("Please, provide an output file")
        return
    output = []
    for i in range(int(1/fraction)):
        training_set = lines[0: i * fr_size] + lines[(i + 1) * fr_size:]
        testing_set = lines[i * fr_size: (i + 1) * fr_size]
        print('\nLoading PPDB...')
        load_simple_PPDB()
        print('Loading OED...')
        load_simple_OED()
        if fr_size != len(lines):
            print('Loading newsela data...')
            process_lines(training_set, Node.add_data)
        Node.contract_data()
        print('Computing...')
        if test_data:
            output.append(test(testing_set))
            print('Run #' + str(i) + ': n, f, c, i = ' + str(output[-1]))
        else:
            output += evaluate(testing_set)
            print('Run #' + str(i) + ' completed')
    if test_data:
        no_diff = sum([x[0] for x in output])
        fail = sum([x[1] for x in output])
        correct = sum([x[2] for x in output])
        incorrect = sum([x[3] for x in output])
        total = no_diff + fail + correct + incorrect
        tried = fail + correct + incorrect
        tested = correct + incorrect
        print("Total: " + str(total) + ". Tried to test: " + str(tried) + " (" +
              str(round(float(tried)/ total * 100, 2)) + "%)")
        print("Managed to test: " + str(tested) + " (" + str(round(float(
            tested) / tried * 100, 2)) + "% out of that tried)")
        print("Correct: " + str(correct) + " (" + str(round(float(
            correct) / tested * 100, 2)) + "% out of that tested)")
    else:
        with open(output_file, 'w') as file:
            file.writelines('\n'.join(output))


if __name__ == "__main__":
    # create_simple_PPDB()
    # create_simple_OED()
    main(0.2, test_data=True)
