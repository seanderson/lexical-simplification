import sys
sys.path.append("../")
import StanfordParse


def leave_sents_only(filename):
    with open(filename) as file:
        lines = file.readlines()
    with open(filename + ".sents", "w") as file:
        for line in lines:
            file.write(line.split('\t')[3].rstrip('\n') + '\n')
    StanfordParse.tag(filename + ".sents")
    with open(filename + ".sents.tagged") as file:
        lines_tagged = file.readlines()
    if len(lines) != len(lines_tagged):
        print("File lengths are unequal!")
        exit(-1)
    for i in range(len(lines)):
        word = lines[i].split('\t')[0].casefold()
        ind = int(lines[i].split('\t')[1].casefold())
        line = lines[i].split('\t')[3].rstrip('\n').casefold().split(' ')
        line_tagged = line_tagged[i].rstrip('\n').casefold().spllit(' ')
        if len(line) != len(line_tagged):
            print(line)
            print(line_tagged)
            print("Line lengths are unequal!")
            exit(-1)
        if word != line[ind]:
            print(word + " " + str(ind))
            print(line[ind])
            print("Inconsistency withing the line!")
            exit(-1)
        if word != line_tagged[ind]:
            print(word + " " + str(ind))
            print(line_tagged[ind])
            print("Inconsistency withing the tagged line!")
            exit(-1)
    print("Check_completed")


if __name__ == "__main__":
    leave_sents_only("/home/nlp/corpora/newsela_complex/Newsela_Complex_Words_Dataset_supplied.txt")