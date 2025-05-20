import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from xgboost import XGBClassifier

# Define the function to analyze sentiment
def analize_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    return analysis.polarity

# Load dataset
news = pd.read_csv(r'C:/Users/tharu/OneDrive/Desktop/Stock_Prediction_Project/Combined_News_DJIA.csv')

# Splitting dataset into train and test
train_news = news[news['Date'] < '2014-07-15']
test_news = news[news['Date'] > '2014-07-14']

# Convert training dataset to list by concatenating columns 2 to 26
train_news_list = [' '.join(str(k) for k in train_news.iloc[row, 2:27]) for row in range(len(train_news))]

# Vectorization with CountVectorizer
vectorize = CountVectorizer(min_df=0.01, max_df=0.8)
news_vector = vectorize.fit_transform(train_news_list)

# Logistic Regression Model
lr = LogisticRegression()
model = lr.fit(news_vector, train_news["Label"])

# Convert testing dataset to list
test_news_list = [' '.join(str(x) for x in test_news.iloc[row, 2:27]) for row in range(len(test_news))]
test_vector = vectorize.transform(test_news_list)

# Predictions
predictions = model.predict(test_vector)
accuracy1 = accuracy_score(test_news['Label'], predictions)
print("Baseline model accuracy:", accuracy1)

# TF-IDF Bigram Vectorizer
nvectorize = TfidfVectorizer(min_df=0.05, max_df=0.85, ngram_range=(2, 2))
news_nvector = nvectorize.fit_transform(train_news_list)
nmodel = lr.fit(news_nvector, train_news["Label"])
ntest_vector = nvectorize.transform(test_news_list)
npredictions = nmodel.predict(ntest_vector)
accuracy2 = accuracy_score(test_news['Label'], npredictions)
print("Logistic Regression with Bigram and TF-IDF Accuracy:", accuracy2)

# Random Forest Classifier
rfmodel = RandomForestClassifier(random_state=55)
rfmodel.fit(news_nvector, train_news["Label"])
rfpredictions = rfmodel.predict(ntest_vector)
accuracyrf = accuracy_score(test_news['Label'], rfpredictions)
print("Random Forest Accuracy:", accuracyrf)

# Naive Bayes
nbmodel = MultinomialNB(alpha=0.5)
nbmodel.fit(news_nvector, train_news["Label"])
nbpredictions = nbmodel.predict(ntest_vector)
nbaccuracy = accuracy_score(test_news['Label'], nbpredictions)
print("Naive Bayes Accuracy:", nbaccuracy)

# Gradient Boosting Classifier
gbmodel = GradientBoostingClassifier(random_state=52)
gbmodel.fit(news_nvector, train_news["Label"])
gbpredictions = gbmodel.predict(ntest_vector.toarray())
gbaccuracy = accuracy_score(test_news['Label'], gbpredictions)
print("Gradient Boosting Accuracy:", gbaccuracy)
print("Confusion Matrix:\n", confusion_matrix(test_news['Label'], gbpredictions))

# Trigram TF-IDF Vectorizer
n3vectorize = TfidfVectorizer(min_df=0.0004, max_df=0.115, ngram_range=(3, 3))
news_n3vector = n3vectorize.fit_transform(train_news_list)
n3model = lr.fit(news_n3vector, train_news["Label"])
n3test_vector = n3vectorize.transform(test_news_list)
n3predictions = n3model.predict(n3test_vector)
accuracy3 = accuracy_score(test_news['Label'], n3predictions)
print("Trigram Accuracy:", accuracy3)

# Sentiment Analysis - apply polarity function to all news columns except Date and Label
train_sentiment = train_news.drop(['Date', 'Label'], axis=1).applymap(analize_sentiment) + 10
test_sentiment = test_news.drop(['Date', 'Label'], axis=1).applymap(analize_sentiment) + 10

# XGBoost Model on sentiment features
XGB_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
XGB_model.fit(train_sentiment, train_news['Label'])
y_pred = XGB_model.predict(test_sentiment)

print("Sentiment Analysis Accuracy:", accuracy_score(test_news['Label'], y_pred))
print("F1 Score:\n", classification_report(test_news['Label'], y_pred))
