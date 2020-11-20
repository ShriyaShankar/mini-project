import pandas as pd 
import numpy as np 
import os
import civic_issues as ci 
# import streamingtweets as st 
import metrics 
from sklearn.model_selection import train_test_split
import pickle

pd.options.mode.chained_assignment = None

def civic_framework(test_df, tweet_or_text = 0):
    
    text_data = ci.preprocess(0, test_df)[['processedtext']]

    text_data = list(text_data.processedtext)

    # print(test_df.head())

    dataset = pd.read_csv("finalData.csv")
    dataset.drop_duplicates(subset='description',inplace=True,keep=False)

    dataset = ci.preprocess(0, dataset)
    data = dataset[['processedtext','civic_issue']]

    # Splitting into test and train
    X_train, X_test, y_train, y_test = train_test_split(data['processedtext'],data['civic_issue'], test_size=0.20, random_state=42)
    # print(list(X_train))
    train_tfidf, test_tfidf = ci.vectorize(0,list(X_train),text_data)

    # logreg_model = ci.logreg_tune(train_tfidf, y_train)

    filename = 'logRegclassifier.sav'
    logreg_model = pickle.load(open(filename, 'rb'))
    prediction = logreg_model.predict(test_tfidf)

    test_df['processedtext'] = text_data
    test_df['civic_issue'] = prediction

    print(test_df[['description','civic_issue']])

    civic_issue_text = test_df[test_df['civic_issue']==1]

    if len(civic_issue_text)==0:
        print("No civic issue tweets/reports to display.")
        # print(civic_issue_text['description'])
        return False

    print("\n\n\nMoving to Further Processing...\n\n\n")
    civic_data = dataset[dataset['civic_issue']==1][['processedtext', 'category']]

    X_train, X_test, y_train, y_test = train_test_split(civic_data['processedtext'],civic_data['category'], test_size=0.20, random_state=42)
    train_tfidf, test_tfidf = ci.vectorize(0, list(X_train), list(civic_issue_text.processedtext))

    # svc_model = ci.train_SVC(train_tfidf, y_train)
    filename = 'linearkernelSVC.sav'
    svc_model = pickle.load(open(filename, 'rb'))
    prediction = svc_model.predict(test_tfidf)

    civic_issue_text['category'] = prediction

    civic_issue_text['sentiment'] = civic_issue_text.description.apply(ci.calculate_sentiment)
    
    if tweet_or_text==1: # Tweets
        print("\n\nCivic Issue Related Tweets\n")
        print(civic_issue_text[['description','username','timestamp','category', 'sentiment']])
    else:
        print("Civic Issue Related Inputs\n")
        print(civic_issue_text[['description','category', 'sentiment']])

    return True

if __name__=="__main__":
    f = open("testdata.txt","r")
    text = [i.rstrip() for i in f.readlines()]

    df_test = pd.DataFrame()
    df_test['description'] = text
    civic_framework(df_test,0)