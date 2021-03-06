"""
Uses gensim to create embedding vectors in which every word has its POS marked.
"""
import math

from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
import pathlib
import datetime
from src.word2vec.globdefs import *
import src.word2vec.prepDataPOS as prepData


MODEl_EXT = ".model"
# For details see https://radimrehurek.com/gensim/models/word2vec.html
SKIPGRAM = 0                # use CBOW
EMBED_SIZE = 200            # length of embedding vectors
WINDOW = 5
# maximum distance between the current and predicted word
ALPHA = 0.01                 # The initial learning rate.
MIN_ALPHA = 1.0e-9          # Learning rate will linearly drop to min_alpha
SEED = 666                   # Seed for the random number generator.
MIN_COUNT = 50
# Ignores all words with total frequency lower than this.
MAX_VOCAB_SIZE = None       # Limits the RAM during vocabulary building
NEGATIVE_SAMPLING = 5       # Use negative sampling
SAMPLE = 0                  # The threshold for configuring which
# higher-frequency words are randomly downsampled
NTHREADS = 8               # number of threads for training
HS = 0
# If 1, hierarchical softmax will be used for model training.
HASHFXN = None
# Hash function to use to randomly initialize weights
ITER = 5                  # Number of iterations (epochs) over the corpus.
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
        output_path = '{}/epoch{}.model'.format(self.path_prefix, self.epoch)
        print("Save model to {}".format(output_path))
        model.save(output_path)
        print(model.running_training_loss)
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


def main(DATABASE=None):
    # output from training in new directory
    mtype = 'skip' if SKIPGRAM == 1 else 'cbow'
    outdir = mtype + '-{:%Y-%b-%d-%H%M}'.format(datetime.datetime.now())
    outdir = "/home/nlp/wpred/word2vecmodels/" + outdir
    pathlib.Path(outdir).mkdir(parents=False, exist_ok=False)

    epoch_saver = EpochSaver(outdir)

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
                              callbacks=[epoch_saver],
                              compute_loss=True,
                              alpha=ALPHA,
                              min_alpha=ALPHA)
    # add vocabulary
    if not DATABASE:
        prepData.build_lexicon(model)
    else:
        prepData.build_lexicon(model, corpus=DATABASE)
    flist = []

    if not DATABASE:
        with open(DATA_METAFILE) as data_meta:
            for line in data_meta:
                with open(line.rstrip('\n')) as database_metafile:
                    for name in database_metafile:
                        flist.append(name.rstrip('\n'))
    else:
        with open("/home/nlp/wpred/text_databases/"+DATABASE+"/metafile.txt") as \
                database_metafile:
                    for name in database_metafile:
                        flist.append(name.rstrip('\n'))
        # with open(flist[0]) as file:
            # sents = file.readlines()
            # for i in range(len(sents)):
                # sents[i] = MySentences.tokenize(sents[i]) + ['\n']
    sent_iter = MySentences(flist)
    model.train(sentences=sent_iter,
                total_examples=sent_iter.countlines(), epochs=ITER, compute_loss=True)


def evaluate(prefix, epochs):
    """
    Evaluate a model on different stages (epochs)
    :param prefix:
    :param epochs:
    :return:
    """
    with open(prefix + "evaluation.txt", "w") as file:
        for epoch in epochs:
            print("Loading epoch " + str(epoch))
            file.write("\nEpoch " + str(epoch) + ":\n")
            model = word2vec.Word2Vec.load(prefix + "epoch" + str(epoch) + MODEl_EXT)

            # test_file = "/home/nlp/newsela/src/nn/SimLex-999.tsv"
            # print(model.wv.evaluate_word_pairs(test_file))
            # file.write(str(model.wv.evaluate_word_pairs(test_file)) + "\n")

            lst = []

            print(model.predict_output_word(['You_P', 'made_V', 'a_DT' 'mistake_N'], topn=50))
            print(model.predict_output_word(
                ['horses_N', 'battle_N', 'king_N' 'killed_V'], topn=50))

            for w0 in ['train_N', 'train_V']:
                lst += [x[0] for x in model.wv.most_similar(positive=[w0], topn=2)]

            lst += ["create_V", "horse_N", "test_V", "test_N", "funny_J", "joke_V", 'train_N', 'train_V', 'once_A']

            for w in lst:
                sim0 = model.wv.similarity('train_N', w)
                sim1 = model.wv.similarity('train_V', w)
                """ x = sim0 / (sim0 + sim1)
                if x > sim0:
                    x = -x / (x - 1)
                y = math.sqrt(sim0 * sim0 - x * x)"""
                print(w, str(round(sim0, 2)), str(round(sim1, 2)))


if __name__ == "__main__":
    # main()
    evaluate("/home/nlp/wpred/word2vecmodels/cbow-2018-Sep-09-2223/", [1])
    # model = word2vec.Word2Vec.load("/home/nlp/wpred/word2vecmodels/cbow-2018-Sep-09-2223/epoch0.model")
    # model = word2vec.Word2VecKeyedVectors.load_word2vec_format("/home/nlp/wpred/word2vecmodels/cbow-2018-Sep-09-2223/model.bin")
    # model.wv.save_word2vec_format("/home/nlp/wpred/word2vecmodels/cbow-2018-Sep-09-2223/model.bin")
    # model.wv.save_word2vec_format("/home/nlp/newsela/src/nn/cbow-2018-Jul-20-1837/model.bin")
