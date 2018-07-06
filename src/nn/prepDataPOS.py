"""
repare data including Part Of Speech
Prepare several corpora found in nlp:corpora/text_databases for
input to gensim/word2vec, which is used to create embedding vectors.
"""

from nn.globdefs import *
import csv
from gensim.models import Word2Vec


def load_vocab(infile=VOCFILE, corpus=VOC_HDR_TOTAL):
    """
    Read raw vocab from corpus into dictionary which is returned.
    The corpus ID is read from the vocab heading line.
    """
    wordfreq = {}
    with open(infile, newline='') as csvfile:
        # '\r' hack to ignore quote characters
        reader = csv.DictReader(csvfile, delimiter='\t', quotechar='\r')

        for row in reader:
            word = row[VOC_HDR_WORD].strip() # strip leading/trailing space
            wordfreq[word] = int(row[corpus])
    return wordfreq


def fixtoken(token):
    token = token.lower()
    try:
        (word, pos) = token.split(POS_DELIM)
    except:
        return WORD_UNKNOWN
    newtoken = POS_DELIM.join((word.casefold(), UTAG_MAP.get(pos, pos)))
    # print(token + " -> " + newtoken)
    return newtoken


def simplifyPOS(wfreq):
    """
    Simplify frequencies by folding together POS tagged items in univeral set.
    Also lowers case via casefold.
    """
    numbadsplit = 0
    wfreq_new = {}
    for token in wfreq:
        try:
            (word, pos) = token.split(POS_DELIM)
        except:
            numbadsplit += 1
            continue  # give up on bad token
        newtag = UTAG_MAP.get(pos, pos)  # default to pos if not in UTAG_MAP
        newtoken = POS_DELIM.join((word.casefold(), newtag))
        wfreq_new[newtoken] = wfreq_new.get(newtoken,0) + wfreq[token]
    print("Num unique tokens %d\tNum bad-tokens %d\n" % (
        len(wfreq_new), numbadsplit))
    return wfreq_new


def build_lexicon(model,infile=VOCFILE,corpus=VOC_HDR_TOTAL):
    """Create lexicon from vocabulary file containing word frequency."""
    wordfreq = load_vocab(infile,corpus)
    wposfreq = simplifyPOS(wordfreq)
    model.build_vocab_from_freq(wposfreq, keep_raw_vocab=True, update=False)


def main():
    model = Word2Vec(sorted_vocab=1,min_count=10)
    # Create and save vocabulary
    build_lexicon(model, corpus=VOC_HDR_SUBTLEX)
    print("voc size %d\n" % len(model.wv.vocab))
    model.save(MODEL_FILE)


if __name__ == '__main__':
    main()
