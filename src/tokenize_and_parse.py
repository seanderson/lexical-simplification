# This is python wrapper around the Stanford tokenizer and parser

import newselautil as nsla
import classpaths as path
import subprocess


MAX_LEVELS = 6  # maximum number of levels in the newsela corpus


def process_all(method):
    """
    Process all articles
    :param method: the method to use to process the files.
    Either tokenize or parse
    :return: None
    """
    articles = nsla.loadMetafile()
    i = 0
    # process articles by slug
    while i < len(articles):
        slug = articles[i]['slug']
        n_of_levels = 1
        while (i < len(articles) - 1) and (articles[i+1]['slug'] == slug):
            n_of_levels += 1
            i += 1
        process_file(slug, method, n_of_levels)
        print('Processed:' + slug + ' '+str(round(i/float(len(articles)), 5)) +
              ' of the task completed')
        i += 1


def process_particular(slugs, method):
    """
    Process with specified slugs
    :param slugs: the list of slugs to tokenize
    :param method: the method to use to process the files
    :return: None
    """
    for slug in slugs:
        process_file(slug, method)


def process_file(slug, method, number_of_levels=6):
    """
    Tokenize all the files with a given slug
    :param slug: the slug to tokenize
    :param number_of_levels: number of levels for this slug
    :return: None
    """
    for i in range(number_of_levels):
        filename = path.NEWSELA + '/articles/' + slug + ".en." + str(i) + ".txt"
        if method == parse:
            filename += ".tok"
        result = method(filename)
        if method == parse:
            with open(filename + ".prs", "w") as file:
                file.writelines(result.decode('utf8'))


def parse(textfile):
    """
    Run parser and return the output as a string.
    :param textfile: the name of the file to process
    :return: None
    """
    output = subprocess.check_output(['java', '-cp', path.CLASSPATH,
                                      '-Xmx8000m', path.PARSERPROG, path.MODELS,
                                      textfile], shell=False)
    return output


def tokenize(textfile):
    """
    Run Stanford tokenizer. Output to textfile.tok
    :param textfile: the name of the file to process
    :return: None
    """
    subprocess.check_output(['java', '-cp', path.CLASSPATH, path.TOKENIZERPROG,
                             textfile], shell=False)


if __name__ == "__main__":
    process_all(tokenize)
