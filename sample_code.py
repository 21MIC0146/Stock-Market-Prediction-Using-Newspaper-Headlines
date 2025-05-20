import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from xgboost import XGBClassifier

def analize_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    return analysis.polarity

# Load dataset
news = pd.read_csv(r'C:/Users/tharu/OneDrive/Desktop/Stock_Prediction_Project/Combined_News_DJIA.csv')

# Split dataset
train_news = news[news['Date'] < '2014-07-15']
test_news = news[news['Date'] > '2014-07-14']

# Combine headlines into one string per row
train_news_list = [' '.join(str(k) for k in train_news.iloc[row, 2:27]) for row in range(len(train_news))]
test_news_list = [' '.join(str(x) for x in test_news.iloc[row, 2:27]) for row in range(len(test_news))]

# Count Vectorizer Baseline
vectorize = CountVectorizer(min_df=0.01, max_df=0.8)
news_vector = vectorize.fit_transform(train_news_list)
lr = LogisticRegression(random_state=42)
model = lr.fit(news_vector, train_news["Label"])
test_vector = vectorize.transform(test_news_list)
predictions = model.predict(test_vector)
print("Baseline model accuracy:", accuracy_score(test_news['Label'], predictions))

# TF-IDF Bigram Vectorizer + Logistic Regression
nvectorize = TfidfVectorizer(min_df=0.05, max_df=0.85, ngram_range=(2,2))
news_nvector = nvectorize.fit_transform(train_news_list)
nmodel = lr.fit(news_nvector, train_news["Label"])
ntest_vector = nvectorize.transform(test_news_list)
npredictions = nmodel.predict(ntest_vector)
print("Logistic Regression with Bigram and TF-IDF Accuracy:", accuracy_score(test_news['Label'], npredictions))

# Random Forest
rfmodel = RandomForestClassifier(random_state=42)
rfmodel.fit(news_nvector, train_news["Label"])
rfpredictions = rfmodel.predict(ntest_vector)
print("Random Forest Accuracy:", accuracy_score(test_news['Label'], rfpredictions))

# Naive Bayes
nbmodel = MultinomialNB(alpha=0.5)
nbmodel.fit(news_nvector, train_news["Label"])
nbpredictions = nbmodel.predict(ntest_vector)
print("Naive Bayes Accuracy:", accuracy_score(test_news['Label'], nbpredictions))

# Gradient Boosting
gbmodel = GradientBoostingClassifier(random_state=42)
gbmodel.fit(news_nvector, train_news["Label"])
gbpredictions = gbmodel.predict(ntest_vector.toarray())
print("Gradient Boosting Accuracy:", accuracy_score(test_news['Label'], gbpredictions))
print("Confusion Matrix:", confusion_matrix(test_news['Label'], gbpredictions))

# Trigram TF-IDF + Logistic Regression
n3vectorize = TfidfVectorizer(min_df=0.0004, max_df=0.115, ngram_range=(3,3))
news_n3vector = n3vectorize.fit_transform(train_news_list)
n3model = lr.fit(news_n3vector, train_news["Label"])
n3test_vector = n3vectorize.transform(test_news_list)
n3predictions = n3model.predict(n3test_vector)
print("Trigram Accuracy:", accuracy_score(test_news['Label'], n3predictions))

# Sentiment Analysis using TextBlob
train_sentiment = train_news.drop(['Date', 'Label'], axis=1).applymap(analize_sentiment)
test_sentiment = test_news.drop(['Date', 'Label'], axis=1).applymap(analize_sentiment)

# Convert to numpy array for XGBoost
X_train_sentiment = train_sentiment.values
X_test_sentiment = test_sentiment.values

# XGBoost Model on sentiment features
XGB_model = XGBClassifier(random_state=42)
gradiant = XGB_model.fit(X_train_sentiment, train_news['Label'])
y_pred = gradiant.predict(X_test_sentiment)
print("Sentiment Analysis Accuracy:", accuracy_score(test_news['Label'], y_pred))
print("F1 Score:", classification_report(test_news['Label'], y_pred))
