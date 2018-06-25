import classpaths as paths
import re
from nltk.stem import WordNetLemmatizer;
Lemmatizer = WordNetLemmatizer()
import newselautil as ns


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
            node1 = get_node(sent_data[i][0], create_new_node=False)
            node2 = get_node(sent_data[j][0], create_new_node=False)
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
                if debug:
                    print("\n\n " + node1.word + "\t" + node2.word + "\t" + str(
                        diff))
                    Node.DEBUG = True
                    Node.calculate_complexity(node1, node2)
                    Node.DEBUG = False
    return fail, correct, incorrect


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
            word = Lemmatizer.lemmatize(word)
            if word not in all_sent:
                sent_data.append((word, 0))
        if train_rather_than_test:
            add_data_to_graph(sent_data)
        else:
            f, c, ic = test_sentence(sent_data)
            fail += f
            correct += c
            incorrect += ic
    if not train_rather_than_test:
        Node.all_nodes = {}
        return fail, correct, incorrect


def add_data_to_graph(sent_data):
    for i in range(len(sent_data)):
        for j in range(i + 1, len(sent_data)):
            diff = sent_data[i][1] - sent_data[j][1]
            if diff > 0:
                get_node(sent_data[i][0]).add(get_node(sent_data[j][0]), diff)
            elif diff < 0:
                get_node(sent_data[j][0]).add(get_node(sent_data[i][0]), diff)


class Node:

    DEBUG = False
    all_nodes = {}

    def __init__(self, word):
        if word in Node.all_nodes:
            print("Warning: the node '" + word + "'already exists")
        Node.all_nodes[word] = self
        self.word = word
        self.edges = {}
        self.count = 0

    def add(self, node, diff):
        self.count += 1
        node.count += 1
        if node not in self.edges:
            self.edges[node] = []
        self.edges[node].append(diff)

    @staticmethod
    def calculate_complexity(node1, node2, scale=1):
        # if node1.count < 5 or node2.count < 5:
            # return None
        if Node.DEBUG:
            print("Analyzing " + node1.word + " to " + node2.word + "-" * 50)
        a = node1.reach(node2, 2)
        if Node.DEBUG:
            print("Analyzing " + node2.word + " to " + node1.word + "-" * 50)
        b = node2.reach(node1, 2)
        if a == b and b == -1:
            return None
        if a > scale * b:
            return 1
        if b > scale * a:
            return -1
        else:
            return 0

    def reach(self, node, deep):
        result = []
        if self == node:
            result = [0]
            if Node.DEBUG:
                print(self.word + " is " + node.word)
        elif deep == 0:
            if node in self.edges:
                if Node.DEBUG:
                    print(self.word + "->" + node.word + " (" + str(round(
                        self.edges[node], 2)) + ")")
                result = [self.edges[node]]
        else:
            for key in self.edges.keys():
                curr = key.reach(node, deep - 1)
                if curr != -1:
                    if Node.DEBUG:
                        print(self.word + "->" + key.word + " (" + str(round(
                            self.edges[key], 2)) + ")")
                    result.append(self.edges[key] + curr)
        if len(result) == 0:
            return -1
        # return float(sum(result)) / len(result)
        return float(sum(result))

    @staticmethod
    def contract_data():
        for name in Node.all_nodes.keys():
            node = get_node(name)
            length = len(node.edges.keys())
            for key in node.edges.keys():
                node.edges[key] = float(sum(node.edges[key]))
                #  node.edges[key] /= length


def get_node(word, create_new_node=True):
    if not re.match('[a-zA-Z]*', word):
        print("WW: Node " + word + " should not be created")
    if word not in Node.all_nodes:
        if create_new_node:
            Node.all_nodes[word] = Node(word)
        else:
            return None
    return Node.all_nodes[word]


def create_simple_ppdb():
    with open(paths.BASEDIR + "/SimplePPDB/simplification-dictionary") as file:
        lines = file.readlines()
    with open(paths.BASEDIR + "/SimplePPDB/simpple-simple-ppdb", 'w') as file:
        for line in lines:
            if ' ' not in line:
                file.write(line)
                continue
            phrase0 = line.split('\t')[-2].split(' ')
            phrase1 = line.rstrip('\n').split('\t')[-1].split(' ')
            word0 = [x for x in phrase0 if x not in phrase1]
            word1 = [x for x in phrase1 if x not in phrase0]
            if not (len(word0) == 1 and len(word1) == 1):
                continue
            if re.match('.*[^a-zA-Z].*', word0[0]) or re.match('.*[^a-zA-Z].*', word1[0]):
                continue
            if word1[0] in ns.STOPWORDS or word1[0] in ns.STOPWORDS:
                continue
            # print(line.rstrip('\n'))
            # print('\t'.join(line.split('\t')[:-2] + word0 + word1 + ["\n"]))
            file.write('\t'.join(line.split('\t')[:-2]
                                 + word0 + [word1[0] + "\n"]))


def load_simple_ppdb():
    with open(paths.BASEDIR + "/SimplePPDB/simpple-simple-ppdb") as file:
        lines = file.readlines()
    for line in lines:
        score1, score2, _, complex, simple = line.split('\t')
        simple = simple.rstrip('\n')
        score = float(score1) * float(score2) * 2
        add_data_to_graph([(complex, score), (simple, 0)])


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
        load_simple_ppdb()
        print('Loading newsela data...')
        process_lines(train, True)
        Node.contract_data()
        print('Testing...')
        fail, correct, incorr = process_lines(test, False)
        print('Run #' + str(i) + ': f, c, i = ' + str([fail, correct, incorr]))


if __name__ == "__main__":
    # create_simple_ppdb()
    main(0.2)

