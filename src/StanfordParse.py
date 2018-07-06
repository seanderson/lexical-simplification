# Run stanford parser ParseDemo on utf-8 text file.
# Be sure to compile ~/lib/standford-parser*/custom/Parser

import subprocess
import sys

import classpaths as path


def parse(textfile):
    """Run parser and return all output as a string."""
    # subprocess.call(['java','-cp',CLASSPATH,PROG,MODELS,textfile],shell=False)
    # output = subprocess.check_output(['java','-cp',CLASSPATH,'-Xmx8000m',,PROG,MODELS,textfile],shell=False)
    output = subprocess.check_output(['java','-cp',path.CLASSPATH,'-Xmx8000m',path.PARSERPROG,path.MODELS,textfile],shell=False)
    return output


def tokenize(textfile):
    """Run tokenizer.  Output in textfile.tok"""
    output = subprocess.check_output(['java','-cp',path.CLASSPATH,path.TOKENIZERPROG,textfile],shell=False)


def tag(textfile):
    print(' '.join((['java', '-mx2048m','-cp',path.CLASSPATH, "edu.stanford.nlp.tagger.maxent.MaxentTagger", "-model", '/home/nlp/newsela/stanford-postagger/models/english-bidirectional-distsim.tagger', "-textFile", textfile])))
    result = subprocess.check_output(['java', '-mx2048m','-cp',path.CLASSPATH, "edu.stanford.nlp.tagger.maxent.MaxentTagger", "-model", '/home/nlp/newsela/stanford-postagger/models/english-bidirectional-distsim.tagger', "-textFile", textfile], shell=False)
    with open(textfile + ".tagged", 'w') as file:
        file.writelines(result)


def tag_all(folder_in, folder_out):
    """
    Read the folder's metadata.txt file and tag all the files listed in it
    :param folder:
    :return:
    """
    with open(folder_in + "/metadata.txt") as file:
        lines = file.readlines()
    for i in range(len(lines)):
        if (float(i) / len(lines) < 0.1):
            continue
        line = "/" + lines[len(lines)-i-1].rstrip('\n')
        print("Tagging: " + folder_in + line)
        output = subprocess.check_output(
            ['java', '-mx7168m', '-cp', path.CLASSPATH,
             "edu.stanford.nlp.tagger.maxent.MaxentTagger", "-model",
             '/home/nlp/newsela/stanford-postagger/models/english-bidirectional-distsim.tagger',
             "-textFile", folder_in + line], shell=False)
        print("Writing " + folder_out + line + ".tagged")
        with open(folder_out + line + ".tagged", 'w') as file:
            file.writelines(output)
        print("Progress: " + str(round(float(i) / len(lines) * 100, 2)) + "%")

def main():
    textfile = sys.argv[1]
    print (parse(textfile))
    # tokenize(textfile)

        
if __name__ == "__main__":
    # main()
    tag_all("/home/nlp/corpora/text_databases",
            "/home/nlp/corpora/text_databases_tagged")
