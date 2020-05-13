import numpy as np
import json
from torch.utils.data import Dataset
import torch as tt
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
tt.manual_seed(seed)


class AlignDataset(Dataset):

    def __init__(self, filename, levels, max_sents, max_words, embedding_size, edge_types,
                 annotation_size=None):
        """
        Initializse the Alignment Dataset by loading data into memory and indexing it
        :param filename:        the .json file from which to load the data
        :param levels:          a tuple of two elements (levels of articles to align)
        :param max_sents:   maximum number of sentences per element in the dataset
        :param max_words:       maximum number of words in a sentence
        :param embedding_size:  size of word embeddings
        :param edge_types:      number of edge types
        :param annotation_size: if not None, this can be greater than embedding size
        """
        with open(filename, 'r') as f:
            self.data = json.load(f)
        self.max_sents = max_sents
        self.max_words = max_words
        self.embedding_size = embedding_size
        self.edge_types = edge_types
        self.levels = [str(x) for x in levels]
        if annotation_size:
            self.annotation_size = annotation_size
        else:
            self.annotation_size = embedding_size
        self.filter_dataset()
        self.id_to_key = list(self.data.keys())

    def filter_dataset(self):
        """
        Remove from the dataset the sentences that are too large. Randomly remove sentences from
        articles that are too large. Also remove articles that do not have the two levels specified
        :return:
        """
        articles = list(self.data.keys())
        for article in articles:
            if self.levels[0] not in self.data[article].keys() or \
                    self.levels[1] not in self.data[article].keys():
                del self.data[article]
                continue
            for level in self.data[article].keys():
                self.data[article][level] = [sent for sent in self.data[article][level]
                                             if len(sent["strings"]) <= self.max_words]
                if len(self.data[article][level]) > self.max_sents:
                    random.shuffle(self.data[article][level])
                    self.data[article][level] = self.data[article][level][:self.max_sents]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        matrix_all = np.zeros((self.max_sents * 2,
                            self.max_words, 
                            self.max_words * self.edge_types * 2), dtype=int)
        features_all = np.zeros((self.max_sents * 2,
                                 self.max_words, self.annotation_size), dtype=float)
        mask = np.zeros(self.max_sents * 2, dtype=int)
        
        for i, sentence in enumerate(self.data[self.id_to_key[index]][self.levels[0]]):
            matrix_all[i], features_all[i] = self.sentence_to_graph(sentence)
            mask[i] = 1
        for i, sentence in enumerate(self.data[self.id_to_key[index]][self.levels[1]]):
            matrix_all[i + self.max_sents], features_all[i + self.max_sents] = self.sentence_to_graph(sentence)
            mask[i + self.max_sents] = 1

        return matrix_all, features_all, mask

    def sentence_to_graph(self, sentence):
        """
        Given a sentence (self.data[i][j]), construct the adjacency matrix and the node
        annotation matrices for that sentence
        :param sentence:    a json object with fields "edges", and "annotations"
        :return:            adjacency-matrix, annotation_matrix
        """
        matrix = self.create_adjacency_matrix(sentence["edges"])
        features = np.pad(sentence['annotations'],
                          ((0, self.max_words - len(sentence['annotations'])),
                           (0, self.annotation_size - self.embedding_size)),
                          'constant')
        return matrix, features

    def create_adjacency_matrix(self, edges):
        """
        Create adjacency matrix given a list of edges (word-to-word relationships)
        :param edges: List of all edges in a graph
        :return: the adjacency matrix
        """
        matrix = np.zeros([self.max_words, self.max_words * self.edge_types * 2])
        for edge in edges:
            src = edge[0]
            e_type = edge[1]
            dest = edge[2]
            self.set_matrix(matrix, src, dest, e_type, 1)
        return matrix

    def set_matrix(self, matrix, src, dest, e_type, value):
        """
        Remove or add an edge in the adjacency matrix. Also remove or add the corresponding edge
        going in the opposite direction
        :param a:     the adjacency matrix
        :param src:   the source node
        :param dest:  the destination node
        :param e_type:the type of the edge to be removed\added
        :param value: 1 if the edge is to be added, 0 otehrwise
        """
        matrix[dest][(e_type - 1) * self.max_words + src] = value
        matrix[src][(e_type - 1 + self.edge_types) * self.max_words + dest] = value


class AlignDatasetTrain(AlignDataset):

    def __init__(self, filename, levels, max_sents, max_words, embedding_size, edge_types,
                 annotation_size=None):
        """
        Initializse the Alignment Dataset by loading data into memory and indexing it
        :param filename:        the .json file from which to load the data
        :param levels:          a tuple of two elements (levels of articles to align)
        :param max_sents:   maximum number of sentences per element in the dataset
        :param max_words:       maximum number of words in a sentence
        :param embedding_size:  size of word embeddings
        :param edge_types:      number of edge types
        :param annotation_size: if not None, this can be greater than embedding size
        """
        super().__init__(filename, levels, max_sents, max_words, embedding_size, edge_types,
                         annotation_size)

    def __len__(self):
        return len(self.data) * 5

    def __getitem__(self, index):
        matrix_all = np.zeros((self.max_sents * 4,
                               self.max_words,
                               self.max_words * self.edge_types * 2), dtype=np.float64)
        features_all = np.zeros((self.max_sents * 4,
                                 self.max_words, self.annotation_size), dtype=np.float64)
        mask = np.ones((self.max_sents * 2, self.max_sents * 2), dtype=np.float64)
        mask[:self.max_sents:, :self.max_sents] = -1
        mask[self.max_sents:, self.max_sents:] = -1

        sts = self.max_sents
        ind1, ind2 = np.random.choice(np.arange(len(self.data)), 2, replace=False)
        matrix_tmp, features_tmp, mask_tmp = super().__getitem__(ind1)
        matrix_all[:sts], features_all[:sts]= matrix_tmp[:sts], features_tmp[:sts]
        matrix_all[sts * 2:sts * 3], features_all[sts * 2:sts * 3] = \
            matrix_tmp[sts:], features_tmp[sts:]
        mask[np.where(mask_tmp[:sts] == 0)[0], :] = 0
        mask[:, np.where(mask_tmp[sts:] == 0)[0]] = 0

        matrix_tmp, features_tmp, mask_tmp = super().__getitem__(ind2)
        matrix_all[sts:sts * 2], features_all[sts:sts * 2] = \
            matrix_tmp[:sts], features_tmp[:sts]
        matrix_all[sts * 3:], features_all[sts * 3:] = matrix_tmp[sts:], features_tmp[sts:]
        mask[np.where(mask_tmp[:sts] == 0)[0] + sts, :] = 0
        mask[:, np.where(mask_tmp[sts:] == 0)[0] + sts] = 0

        return matrix_all, features_all, mask


if __name__ == "__main__":
    dataset = AlignDatasetTrain(filename="/home/nlp/wpred/newsela/articles/parsedSmall.json",
                                levels=(0, 2), max_sents=100, max_words=50, embedding_size=8,
                                edge_types=2)
    test = dataset.__getitem__(0)
    print(test)