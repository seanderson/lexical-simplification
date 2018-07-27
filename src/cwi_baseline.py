import generate_features as gf
import classpaths as paths
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

flist = ["News_Dev.tsv","News_Train.tsv","News_test.tsv",
         "WikiNews_Dev.tsv","WikiNews_Train.tsv","WikiNews_Test.tsv",
         "Wikipedia_Dev.tsv","Wikipedia_Train.tsv","Wikipedia_Test.tsv"]


def train_classifier(x,y):
    clf = NearestCentroid(shrink_threshold=1.)
    clf.fit(x,y)
    return clf

def test_classifier(clf,x,y,multiword=[]):
    pred = clf.predict(x)
    for i in range(len(multiword)):
        if multiword[i] == 1:
            pred[i] = 1 # force multiword to complex

    print precision_recall_fscore_support(y,pred, average='macro')
    print f1_score(y,pred)
    print classification_report(y,pred)

def load_dataset(num):
    f =  flist[num]
    # create directory if necessary
    root = f[:-4].lower()
    feature_file = "features_" + root
    print("loading",gf.FEATURE_DIR+feature_file)
    fe = gf.CustomFeatureEstimator(["word_syllab", "POS", "sent_syllab", "hit",
                                    "1-gram", "word_count", "mean_word_length",
                                    "synset_count", "synonym_count", "labels",
                                    "wv", "2-gram", "vowel_count",
                                    "lexicon"], paths.CWI + feature_file + "/")
    features = fe.load_features()
    labels = fe.load_labels()
    return (features,labels)

def fix_data(features,labels,scaler=None):
    '''Fix up details of the features:
    0. Note all multi-word targets.
    1. Convert hit/count to hit-frequency.
    2. Set 5-gram zeros to 1.
    3. Normalize 5-grams by total count (pseudo-freq)
    4. z-score scale all features.
    '''
    # Convert hit (features 3+4) to a frequency
    hitfreq = features[:,[3]] / features[:,[4]]
    features = np.delete(features,4,1) # remove hit-ttl-count
    multi_word = [ ]
    ismultiword = np.zeros(len(labels))
    sum5gram = 0
    for i in range(len(features)): # remove multi-word targets
        features[i][3] = hitfreq[i]
        if features[i][0] < 1:
            multi_word.append(i)
            ismultiword[i] = 1 # flag it as multiword
        if int(features[i][5]) == 0: # set zero 5-grams to count=1
            features[i][5] = 1
        sum5gram += features[i][5]
    sum5gram = float(sum5gram)
    for i in range(len(features)): # normalize 5-gram
        features[i][5] /= sum5gram
    if scaler == None:
        scaler=StandardScaler()
        scaler.fit(features)
    features = scaler.transform(features)
    oneword_features = np.delete(features,multi_word,0)
    oneword_labels = np.delete(labels,multi_word,0)
    return (scaler,features,labels,oneword_features,oneword_labels,ismultiword)

def main():
    for idx in [0,3,6]:
        print("\n\nDataset %s" % flist[idx])
        scaler = None
        (ftrs,lbls) = load_dataset(idx+1)
        (ftrsdev,lblsdev) = load_dataset(idx+0)
        ftrs = np.concatenate((ftrs,ftrsdev))
        lbls = np.concatenate((lbls,lblsdev))

        (scaler,ftrs,lbls,oftrs,olbls,mword) = fix_data(ftrs,lbls,scaler)

        print("Num targets %d" % len(lbls))
        print("Num removed multi-word targets %d"% (len(lbls)-len(olbls)))
        #print features
        #data = get_cwi_data()
        #result.append(numpy.load(FEATURE_DIR + feature["name"] + ".npy"))

        clf_o = train_classifier(oftrs,olbls) # model for one-word only
        clf = train_classifier(ftrs,lbls) # model for all targets
        # one word targets
        print("-----------------TRAIN--------------------")
        print("---------------All targets------")
        test_classifier(clf,ftrs,lbls,multiword=mword)
        print("---------------One-word targets------")
        test_classifier(clf_o,oftrs,olbls)

        # testing data
        (ftrs,lbls) = load_dataset(idx+2)
        (scaler,ftrs,lbls,oftrs,olbls,mword) = fix_data(ftrs,lbls,scaler)

        print("-----------------TEST--------------------")
        print("Num targets %d" % len(lbls))
        print("Num removed multi-word targets %d"% (len(lbls)-len(olbls)))

        print("---------------All targets------")
        test_classifier(clf,ftrs,lbls,multiword=mword)
        print("---------------One-word targets------")
        test_classifier(clf_o,oftrs,olbls)


if __name__ == "__main__":
    main()
