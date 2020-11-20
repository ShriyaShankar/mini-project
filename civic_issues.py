import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import SVC
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from metrics import model_metrics, auc_roc

dataset = pd.read_csv("finalData.csv")
dataset = dataset[dataset['civic_issue']==1]
dataset = dataset[['description','category']]
dataset.drop_duplicates(subset='description',inplace=True,keep=False)
# dataset.count()

## Preprocessing the Description 
 
#  The preprocessing is done in 4 steps:

#     - removing punctuation
#     - removing stopwords like 'the', 'this','as',etc
#     - conversion of the entire text to lower case
#     - Stemming: reducing the number of inflectional forms of words by reducing all to their common stem.For example, 'argue','arguing','argued' are all reduced to 'argu'
#     - Splitting dataset into train and cross validation sets

def preprocess():
    stemmer = PorterStemmer()
    words = stopwords.words("english")
    dataset['processedtext'] = dataset['description'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

def vectorize(X_train, X_test):
    tfidf = TfidfVectorizer(stop_words='english')
    train_tfidf = tfidf.fit_transform(X_train.values.astype('U'))
    test_tfidf = tfidf.transform(X_test.values.astype('U'))
    return (train_tfidf, test_tfidf)

def train_SVC(train_tfIdf, y_train):
    #building text classification model using Linear Kernel SVC Classifier (has highest accuracy)
    classifier = SVC(kernel='linear') #accuracy obtained for linear kernel = 83.28%
    classifier.fit(train_tfIdf, y_train) #fitting the classifier onto the training data
    filename = "linearkernelSVC.sav"
    pickle.dump(classifier,open(filename,"wb"))

def predict_cat(X_train, X_test, y_train, y_test):  
    # X_train: description data for training
    # y_train: corresponding categories for training
    # X_test and y_test: description and category for testing
    
    # Vectorizing the train and test data using TfIDf vectorization
    # TfIdf - Text Frequency Inverse Document Freqeuncy : vectorizes based on frequency across the current text document but less frequency across multiple documents

    train_tfIdf, test_tfIdf = vectorize(X_train, X_test)
    
    # train_SVC(train_tfIdf, y_train)

    classifier = pickle.load(open("linearkernelSVC.sav","rb"))
    predictions = classifier.predict(test_tfIdf) #predictions made on the unseen data
    train_score = classifier.score(train_tfIdf, y_train)
    print("\n\nTrain Accuracy:",train_score*100,"%\n\n")
    score = classifier.score(test_tfIdf,y_test)
    model_metrics(classifier,y_test,predictions,score)

# predict_cat()

def logReg(train_tfidf, test_tfidf,y_train, y_test, _C=1.0):
    classifier = LogisticRegression(C=_C)
    classifier.fit(train_tfidf,y_train)
    score = classifier.score(test_tfidf,y_test)
    predictions = classifier.predict(test_tfidf)
    model_metrics(classifier,y_test,predictions,score)

def calculate_sentiment(text):
    sentimentAnalyser = SentimentIntensityAnalyzer()
    sentiment = sentimentAnalyser.polarity_scores(text)
    if sentiment['compound'] <= 0.05:
        return ('Negative', 'Complaint')
    elif sentiment['compound'] > 0.05 and sentiment['compound'] <= 0.25:
        return ('Neutral', 'Feedback')
    else:
        return ('Positive','Compliment')