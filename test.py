import pandas as pd 
import numpy as np 
import os
import civic_issues as ci 
import streamingtweets as st 
import metrics 
from sklearn.model_selection import train_test_split
import pickle

f = open("testdata.txt","r")
text = [i.rstrip() for i in f.readlines()]

test_df = pd.DataFrame()
test_df['description'] = text
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

logreg_model = ci.logreg_fit(train_tfidf, y_train)

filename = 'logRegclassifier.sav'
pickle.load(open(filename, 'rb'))
prediction = logreg_model.predict(test_tfidf)

civic_issue_text = pd.DataFrame()
civic_issue_text['processedtext'] = text_data
civic_issue_text['prediction'] = prediction

civic_issue_text = civic_issue_text[civic_issue_text['prediction']==1]

for i in range(len(test_df.description)):
    print(f"Text: {test_df.description[i]}\tCivic Issue: {prediction[i]}")

print("\n\n\nMoving to Further Processing...\n\n\n")
civic_data = dataset[dataset['civic_issue']==1][['processedtext', 'category']]

train_tfidf, test_tfidf = ci.vectorize(0, list(X_train), list(civic_issue_text.processedtext))

svc_model = ci.train_SVC(train_tfidf, y_train)
filename = 'linearkernelSVC.sav'
pickle.load(open(filename, 'rb'))
prediction = svc_model.predict(test_tfidf)

civic_issue_text['category'] = prediction

print(civic_issue_text.head(5))