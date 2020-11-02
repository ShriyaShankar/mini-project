# Civic Tweet Categorisation
### Analysing Civic Issue Complaints made by Citizens on Twitter
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

To analyse real-time data obtained from social media and classify them into civic issue categories, in order to understand citizen proactivity on social media platforms and help improve responsiveness of local government by making categorised data accessible.

### Dataset
- Reap Benefit's Neighbourhood Dashboard data
- Janaagraha data
- Tweets scraped using Tweepy and GetOldTweets3

Hyperlocal tweets scraped using the Twitter Streaming API are preprocessed and passed to a binary Logistic Regression classifier, which classifies the tweet as a civic issue or a non-civic issue; the latter being filtered out. The civic issue tweets are then passed to a Linear kernel Support Vector machine which categorises the tweets into predefined categories like Waste/Garbarge related, Potholes, Water, Sanitation, etc. Sentiment analysis is finally performed on these tweets using TextBlob to determine whether the tweets are complaints requiring urgent attention, neutral feedback or compliments to the authorities. 

Implemented using Python.

### Authors:
- Shriya Shankar
- Adithi Satish
