#File: sentiment_mod.py

import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

import time
localTime = time.asctime(time.localtime(time.time()))
print(" -------------- ENTRY AT " + str(localTime) + " ---------------- ")

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#region ----------- Loading Features, Doccuments & Sets ------------
# - - - - File Path
filePath = "MLSentAnalysis/Trained_Algos/" 
# - - - Loading Doccuments
documents_f = open(filePath + "documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

# - - - Loading Features
word_features5k_f = open(filePath + "word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

# - - - Loading Feature Sets
featuresets_f = open(filePath + "featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()
#endregion -------- Loading Features, Doccuments & Sets ------------


# - - Randomizing input order
random.shuffle(featuresets)

# - - Traing and testing data split
testing_set = featuresets[10000:]
training_set = featuresets[:10000]

#region ========= Loading Pre-Traind algorithms x6 ===============
# ------------ 1/6  NaiveBayes "Control Algorithm"-----------
open_file = open(filePath + "originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()
# ------------ 2/6 MNB_classifier Classifier ----------------
open_file = open(filePath + "MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()
# ------------ 3/6 BernoulliNB Classifier -------------------
open_file = open(filePath + "BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
# ------------ 4/6 Logistic Regression Classifier -----------
open_file = open(filePath + "LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()
# ------------ 5/6 SGDClassifier Classifier -----------------
open_file = open(filePath + "SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()
# ------------ 6/6 LinearSVC Classifier ---------------------
open_file = open(filePath + "LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()
#endregion

# - - - - - - Creating "sertainty" of the system - - - - - - -
voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)





# --- OutPut Function("Text to Be analysied" ----)
def sentiment(text):
    feats = find_features(text)
    sentiment = voted_classifier.classify(feats),voted_classifier.confidence(feats)
    # - - - - - Section takes tuple and creates 1 numeric output - - - - - - 
    if sentiment[0] == "neg":
        num_sentiment = -1
        num_sentiment = num_sentiment * sentiment[1]
    elif sentiment[0] == "pos":
        num_sentiment = 1
        num_sentiment = num_sentiment * sentiment[1]
    else:
        num_sentiment = 0
    return num_sentiment


