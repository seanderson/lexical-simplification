# coding=utf-8
# some basic utilities

import nltk
from nltk.corpus import wordnet
import re


lemmatizer = nltk.WordNetLemmatizer()

re_weblink = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
re_email = re.compile('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+')
re_ip = re.compile('(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})')
re_hash = re.compile(r"([a-fA-F\d]{40})")
re_user = re.compile('(@[a-zA-Z]+)')

bad_chars = "@<>$\"#%&()\'*+-=[]^_`{|}~“”—/"  # chars to be removed

re_is = re.compile("(['`]s)\s")
re_are = re.compile("(['`]re)\s")
re_have = re.compile("(['`]ve)\s")
re_not = re.compile("(n['`]t)\s")
re_am = re.compile("(['`]m)\s")

re_punct = re.compile("([.,?!;:])")


def wordnet_tag(treebank_tag):
    """Convert treebank POS tag (returned by NLTK tagger) to a wordnet POS tag (used by NLTK
    lemmatizer) """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return None


# list of all the treebank POS tags:
POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
       'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
       'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
       'WDT', 'WP', 'WP$', 'WRB']
POS = {tag: wordnet_tag(tag) for tag in POS}


def tokenize(text):
    """
    Get some text as a string, clean it, and return a list of tokens
    :param text: a string
    :return:     list of tokens
    """
    text = re_weblink.sub(' weblink ', text)
    text = re_email.sub(' email ', text)
    text = re_ip.sub(' IP ', text)
    text = re_hash.sub(' hash ', text)
    text = re_user.sub(' user ', text)
    text = re_is.sub(' is ', text)
    text = re_are.sub(' are ', text)
    text = re_have.sub(' have ', text)
    text = re_not.sub(' not ', text)
    text = re_am.sub(' am ', text)
    text = re_punct.sub(' \\1 ', text)

    for p in bad_chars:  # some of these have to be escaped, so it is safer not to use regex
        text = text.replace(p, ' ')

    text = ' '.join([w for w in text.split(' ')])
    text = re.sub('[\n\t ][\n\t ]*', ' ', text).strip(' \t\n')  # strip excessive whitespaces

    return text.strip().split(' ')


def lemmatize(sent):
    """
    Get a list of tokens and return a list of lemmas
    :param sent: a list of tokens
    :return:     a dictionary for lemmas
    """
    tokens = [x.lower() for x in sent]
    pos_tags = [x[1] for x in nltk.pos_tag(tokens)]
    lemmas = {x: x for x in tokens}
    for i in range(len(tokens)):
        if tokens[i].isalpha() and POS.get(pos_tags[i], None) is not None:
            lemmas[tokens[i]] = lemmatizer.lemmatize(tokens[i], POS[pos_tags[i]])
    return lemmas
