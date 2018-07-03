"""
Uses gensim to create embedding vectors in which every word has its POS marked.
"""

from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
import datetime
from nn.globdefs import *
import nn.prepDataPOS as prepData


# For details see https://radimrehurek.com/gensim/models/word2vec.html
SKIPGRAM = 0                # use CBOW
EMBED_SIZE = 500            # length of embedding vectors
WINDOW = 5
# maximum distance between the current and predicted word
ALPHA = 0.1                 # The initial learning rate.
MIN_ALPHA = 1.0e-9          # Learning rate will linearly drop to min_alpha
SEED = 13                   # Seed for the random number generator.
MIN_COUNT = 50
# Ignores all words with total frequency lower than this.
MAX_VOCAB_SIZE = None       # Limits the RAM during vocabulary building
NEGATIVE_SAMPLING = 1       # Use negative sampling
SAMPLE = 0                  # The threshold for configuring which
# higher-frequency words are randomly downsampled
NTHREADS = 10               # number of threads for training
HS = 0
# If 1, hierarchical softmax will be used for model training.
HASHFXN = None
# Hash function to use to randomly initialize weights
ITER = 3                    # Number of iterations (epochs) over the corpus.
TRIM_RULE = None            # Vocabulary trimming rule
SORTED_VOCAB = 1            # If 1, sort the vocabulary by descending frequency
# before assigning word indexes.
BATCH_WORDS = 10000
# Target size (in words) for batches of examples passed to worker threads
COMPUTE_LOSS = True
# If True, computes and stores loss value which can be retrieved.


class EpochSaver(CallbackAny2Vec):
    """
    Callback to save model after every epoch
    This class comes with gensim documentation
    """

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        now = datetime.datetime.now()
        date = '.'.join([str(now.month), str(now.day), str(now.hour),
                         str(now.minute)])
        output_path = '{}_version{}_epoch{}.model'.format(self.path_prefix,
                                                          date, self.epoch)
        print("Save model to {}".format(output_path))
        model.save(output_path)
        self.epoch += 1


class MySentences(object):
    """
    Load raw data into gensim.
    """

    def __init__(self, filelist):
        self.filelist = filelist

    def __iter__(self):
        """
        Iterator to provide sentences to gensim word2vec.
        """
        for fname in self.filelist:
            for line in open(fname, 'r'):
                yield self.tokenize(line)

    def countlines(self):
        """Return the total number of lines in all files."""
        result = 0
        for fname in self.filelist:
            with open(fname) as f:
                for i, l in enumerate(f):
                    pass
            result += i + 1
        return result

    @staticmethod
    def tokenize(line):
        return [prepData.fixtoken(w) for w in line.split()]


def main():
    now = datetime.datetime.now()
    date = '.'.join([str(now.year), str(now.month), str(now.day)])
    epoch_saver = EpochSaver("/home/nlp/newsela/src/w2v" + date)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    # create empty model
    model = word2vec.Word2Vec(iter=1,
                              min_count=MIN_COUNT,
                              sg=SKIPGRAM,
                              hs=HS,
                              negative=NEGATIVE_SAMPLING,
                              size=EMBED_SIZE,
                              workers=NTHREADS,
                              window=WINDOW,
                              callbacks=[epoch_saver])
    # add vocabulary
    prepData.build_lexicon(model)
    flist = []

    with open(DATA_METAFILE) as data_meta:
        for line in data_meta:
            with open(line.rstrip('\n')) as database_metafile:
                for name in database_metafile:
                    flist.append(name.rstrip('\n'))
    """
    with open("/home/nlp/corpora/text_databases/Subtlex/metafile.txt") as \
            database_metafile:
                for name in database_metafile:
                    flist.append(name.rstrip('\n'))
    """
    sent_iter = MySentences(flist)
    model.train(sentences=sent_iter,
                total_examples=sent_iter.countlines(), epochs=ITER)

    # save and reload the model
    mtype = 'skip' if SKIPGRAM == 1 else 'cbow'
    model.save("%s-%d.mdl" % (mtype, EMBED_SIZE))


if __name__ == "__main__":
    main()
