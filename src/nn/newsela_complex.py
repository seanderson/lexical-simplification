import sys

from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
sys.path.append("../")
import StanfordParse
import re

PREFIX = "SDGHKASJDGHKJA"

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
        self.epoch += 1

def process(filename, model, output_name, emb_size):
    with open(filename) as file:
        lines = file.readlines()
    """with open(filename + ".sents", "w") as file:
        for line in lines:
            file.write(PREFIX + " " + line.split('\t')[3].rstrip('\n') + '\n')
    StanfordParse.tag(filename + ".sents")"""
    with open(filename + ".sents.tagged") as file:
        lines_tagged = file.readlines()
    j = 0
    while j < len(lines_tagged):
        if lines_tagged[j][:len(PREFIX)] == PREFIX:
            lines_tagged[j] = ' '.join(lines_tagged[j].split(' ')[1:])
            j += 1
        else:
            lines_tagged[j-1] = lines_tagged[j-1].rstrip(' \n') + ' ' + lines_tagged[j]
            del lines_tagged[j]
    if len(lines) != len(lines_tagged):
        print("File lengths are unequal!")
        exit(-1)
    final_lines = []
    for i in range(len(lines)):
        word = lines[i].split('\t')[0].casefold()
        ind = int(lines[i].split('\t')[1].casefold())
        line = lines[i].split('\t')[3].rstrip('\n').casefold()
        if re.match('.*f331e.s3.amazonaws.com.*?&gt ; .*', line) and ind > 0:
            ind -= 23
        line = re.sub('.*f331e.s3.amazonaws.com.*?&gt ; ', '', line).split(' ')
        line_tagged = lines_tagged[i].rstrip('\n').casefold()
        line_tagged = re.sub('.*f331e.s3.amazonaws.com.*?&_cc gt_nn ;_: ', '', line_tagged)
        line_tagged = re.sub('a\.m\._nn \._\.', 'a.m._nn',
                             line_tagged)
        line_tagged = re.sub('u\.s_nnp \._\.', 'u.s._nnp', line_tagged).split(' ')
        if '\xa0' in line:
            line_tagged.insert(line.index('\xa0'), '\xa0')
        if len(line) != len(line_tagged):
            print(line)
            print(line_tagged)
            print("Line lengths are unequal! ln:" + str(i))
            exit(-1)
        if word != line[ind]:
            if word == line[ind - 2]:
                ind -= 2
            else:
                print(word + " " + str(ind))
                print(line[ind])
                print("Inconsistency withing the line!" + str(i))
                exit(-1)
        if word != '_'.join(line_tagged[ind].split('_')[:-1]):
            if re.sub('&amp;', '&', word) != '_'.join(line_tagged[ind].split('_')[:-1]):
                print(word + " " + str(ind))
                print(line_tagged[ind])
                print("Inconsistency withing the tagged line!")
                exit(-1)
        final_lines.append(lines_tagged[ind])
    print("Check_completed")
    model = word2vec.Word2Vec.load(model)
    print("Model_loaded")
    with open(output_name, 'w') as out_file:
        for word in final_lines:
            if word not in model.wv.vocab:
                print("WW: word not in vocabulary: " + word)
                vector = "0\t" * (emb_size-1) + "0"
            else:
                vector = '\t'.join(model.wv.get_vector(word))
            out_file.write(word + '\t' + vector + '\n')


if __name__ == "__main__":
    process("/home/nlp/corpora/newsela_complex/Newsela_Complex_Words_Dataset_supplied.txt",
                    "/home/nlp/newsela/src/nn/cbow-2018-Jul-05-1347/epoch4.model",
                     "/home/nlp/corpora/newsela_complex/word_embeddings_Jul-05-1256_epoch0.tsv",
            1300)