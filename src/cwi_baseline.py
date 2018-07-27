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

def test_classifier(clf,x,y):
    pred = clf.predict(x)
    print precision_recall_fscore_support(y,pred, average='macro')
    print f1_score(y,pred)
    print classification_report(y,pred)

def load_dataset(num):
    f =  flist[num]
    # create directory if necessary
    root = f[:-4].lower()
    feature_file = "features_" + root
    gf.FEATURE_DIR = paths.CWI + feature_file + "/"
    print("loading",gf.FEATURE_DIR+feature_file)
    fe = gf.CustomFeatureEstimator([
        "labels",
        "word_length","vowel_count","word_syllab",
        "hit" ,"SEW_freq" , "5-gram"
    ])
    features = fe.load_features()
    labels = fe.load_labels()
    return (features,labels)

def fix_data(features,labels,scaler=None):
    # Convert hit (features 3+4) to a frequency
    hitfreq = features[:,[3]] / features[:,[4]]
    features = np.delete(features,4,1) # remove hit-ttl-count
    multi_word = [ ]
    fixedlabels = np.copy(labels) # assume multi-word => complex
    for i in range(len(features)): # remove multi-word targets
        features[i][3] = hitfreq[i]
        if features[i][0] < 1:
            multi_word.append(i)
            fixedlabels[i] = 1 # complex
    if scaler == None:
        scaler=StandardScaler()
        scaler.fit(features)
    features = scaler.transform(features)
    oneword_features = np.delete(features,multi_word,0)
    oneword_labels = np.delete(labels,multi_word,0)
    return (scaler,features,labels,oneword_features,oneword_labels,fixedlabels)

def main():
    scaler = None
    (ftrs,lbls) = load_dataset(1)
    (ftrsdev,lblsdev) = load_dataset(0)
    ftrs = np.concatenate((ftrs,ftrsdev))
    lbls = np.concatenate((lbls,lblsdev))

    (scaler,ftrs,lbls,oftrs,olbls,fixedlbls) = fix_data(ftrs,lbls,scaler)


    print("Num targets %d" % len(lbls))
    print("Num removed multi-word targets %d"% (len(lbls)-len(olbls)))
    #print features
    #data = get_cwi_data()
    #result.append(numpy.load(FEATURE_DIR + feature["name"] + ".npy"))
    clf = train_classifier(oftrs,olbls)
    test_classifier(clf,oftrs,olbls)
    test_classifier(clf,ftrs,fixedlbls)
    # testing data
    (ftrs,lbls) = load_dataset(2)
    (scaler,ftrs,lbls,oftrs,olbls,fixedlbls) = fix_data(ftrs,lbls,scaler)

    print("-----------------TEST--------------------")
    print("Num targets %d" % len(lbls))
    print("Num removed multi-word targets %d"% (len(lbls)-len(olbls)))
    test_classifier(clf,oftrs,olbls)
    test_classifier(clf,ftrs,fixedlbls)


if __name__ == "__main__":
    main()
