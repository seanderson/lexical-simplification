import copy

import src.cwi.generate_features as gf
import src.classpaths as paths
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

cwi = True
wv = True

flist = ["News_Dev.tsv","News_Train.tsv","News_Test.tsv",
         "WikiNews_Dev.tsv","WikiNews_Train.tsv","WikiNews_Test.tsv",
         "Wikipedia_Dev.tsv","Wikipedia_Train.tsv","Wikipedia_Test.tsv"]


def train_classifier(x,y):

    print("training")
    # clf = NearestCentroid(shrink_threshold=1.)
    clf = AdaBoostClassifier(
        RandomForestClassifier(max_depth=2,max_features=None),
        n_estimators = 500,
        learning_rate=1)
    clf.fit(x,y)
    return clf

def test_classifier(clf,x,y,multiword=[]):
    pred = clf.predict(x)
    for i in range(len(multiword)):
        if multiword[i] == 1:
            pred[i] = 1 # force multiword to complex

    print(precision_recall_fscore_support(y,pred, average='macro'))
    print(f1_score(y,pred))
    print(classification_report(y,pred))


def load_dataset(num):
    f =  flist[num]
    # create directory if necessary
    root = f[:-4]
    if (num + 1) % 3 == 0:
        feature_file = "testset/english/" + root + "_Features"
    else:
        feature_file = "traindevset/english/" + root + "_Features"
    print(feature_file)
    feature_names = [
        "labels",
        "word_length","vowel_count","word_syllab",
        "hit" ,"1-gram", "5-gram", "POS", "2-gram", "3-gram", "4-gram",
        "sent_syllab", "word_count", "lexicon", "mean_word_length", "punctuation"
    ]
    if wv:
        feature_names += ["wv"]
    fe = gf.CustomFeatureEstimator(feature_names, "/home/nlp/wpred/datasets/cwi/" + feature_file + "/")
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
    5. Keep only last column in 5-grams.
    '''
    # Convert hit (features 3+4) to a frequency
    multi_word = [ ]
    ismultiword = np.zeros(len(labels))
    for i in range(len(features)): # remove multi-word targets
        if features[i][0] < 1:
            multi_word.append(i)
            ismultiword[i] = 1 # flag it as multiword
    if wv:
        if scaler == None:
            scaler=StandardScaler()
            scaler.fit([f[:-500] for f in features])
        oldfeatures = copy.deepcopy(features)
        features = scaler.transform([f[:-500] for f in oldfeatures])
        features = [np.concatenate((features[i], oldfeatures[i][-500:])) for i in range(len(features))]
    else:
        if scaler == None:
            scaler = StandardScaler()
            scaler.fit(features)
        features = scaler.transform(features)
    oneword_features = np.delete(features,multi_word,0)
    oneword_labels = np.delete(labels,multi_word,0)
    return (scaler,features,labels,oneword_features,oneword_labels,ismultiword)



def main():
    if cwi:
        for idx in [0,3,6]:
            print("\n\nDataset %s" % flist[idx])
            scaler = None

            (ftrs,lbls) = load_dataset(idx+1)
            (ftrsdev,lblsdev) = load_dataset(idx+0)

            print( "shapes",ftrs.shape,ftrsdev.shape)
            ftrs = np.concatenate((ftrs,ftrsdev))
            lbls = np.concatenate((lbls,lblsdev))

            (scaler,ftrs,lbls,oftrs,olbls,mword) = fix_data(ftrs,lbls,scaler)

            print("Num targets %d" % len(lbls))
            print("Num removed multi-word targets %d"% (len(lbls)-len(olbls)))
            #print features
            #data = get_cwi_data()
            #result.append(numpy.load(FEATURE_DIR + feature["name"] + ".npy"))

            # clf_o = train_classifier(oftrs,olbls) # model for one-word only
            clf = train_classifier(ftrs,lbls) # model for all targets
            # one word targets
            """print("-----------------TRAIN--------------------")
            print("---------------All targets------")
            test_classifier(clf,ftrs,lbls,multiword=mword)
            print("---------------One-word targets------")
            test_classifier(clf_o,oftrs,olbls)"""

            # testing data
            (ftrs,lbls) = load_dataset(idx+2)
            (scaler,ftrs,lbls,oftrs,olbls,mword) = fix_data(ftrs,lbls,scaler)

            print("-----------------TEST--------------------")
            print("Num targets %d" % len(lbls))
            print("Num removed multi-word targets %d"% (len(lbls)-len(olbls)))

            print("---------------All targets------")
            test_classifier(clf,ftrs,lbls,multiword=mword)
            """print("---------------One-word targets------")
            test_classifier(clf_o,oftrs,olbls)"""
    else:
        fe = gf.CustomFeatureEstimator([
            "labels",
            "word_length", "vowel_count", "word_syllab", "1-gram", "5-gram", "POS",
            "sent_syllab", "word_count", "lexicon", "mean_word_length", "punctuation",
        ], "/home/nlp/wpred/datasets/kriz/")
        features = fe.load_features()
        labels = fe.load_labels()

        test =  int(len(features) / 10)

        for i in range(10):
            testing_set = [features[i * test:(i + 1) * test], labels[i * test:(i + 1) * test]]
            training_set = [np.concatenate((features[:i * test], features[(i + 1) * test:])), np.concatenate((labels[:i * test], labels[(i + 1) * test:]))]
            scaler = StandardScaler()
            scaler.fit(training_set[0])
            training_set[0] = scaler.transform(training_set[0])
            testing_set[0] = scaler.transform(testing_set[0])
            clf = train_classifier(training_set[0], training_set[1])  # model for all targets
            test_classifier(clf, testing_set[0], testing_set[1])





if __name__ == "__main__":
    main()