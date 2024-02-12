import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from tensorflow import keras

# download Punkt Sentence Tokenizer
nltk.download('punkt')
# download stopwords
nltk.download('stopwords')
from google.colab import drive
drive.mount('/content/drive')

csv_path = '/content/drive/MyDrive/news.csv'

# Read the Excel file into a pandas DataFrame
df = pd.read_csv(csv_path)

df.head()

df = df[['title','text','label']]
df.head()
df['Combined_Text'] = df['title'] + ' ' + df['text']

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

english_stopwords = stopwords.words('english')
stemmer = PorterStemmer()

# define cleaning function
def clean_review(text):
  # convert to lower case
  text = text.lower()

  # remove non alphabetic characters ^
  text = re.sub(r'[^a-z]', ' ', text)

  # stem words
  # tokenize sentences
  tokens = word_tokenize(text)

  # Porter Stemmer
  stemmed = [stemmer.stem(word) for word in tokens]

  # reconstruct the text
  text = ' '.join(stemmed)

  # remove stopwords
  text = ' '.join([word for word in text.split() if word not in english_stopwords])

  return text

print(df['Combined_Text'][1])
print(clean_review(df['Combined_Text'][1]))

df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
df['clean_review'] = df['Combined_Text'].apply(clean_review)
df.head()

from sklearn.model_selection import train_test_split

X = df['clean_review'].values
y = df['label'].values

# Split data into 50% training & 50% test
# Use a random state of 42 for example to ensure having the same split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer

# define a CountVectorizer (with binary=True and max_features=10000)
vectorizer = CountVectorizer(binary = True, max_features = 10000)

# learn the vocabulary of all tokens in our training dataset
vectorizer.fit(x_train)

# transform x_train to bag of words
x_train_bow = vectorizer.transform(x_train)
x_test_bow = vectorizer.transform(x_test)

print(x_train_bow.shape, y_train.shape)
print(x_test_bow.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

# define the LogisticRegression classifier
model = LogisticRegression()

# train the classifier on the training data
model.fit(x_train_bow, y_train)

# get the mean accuracy on the training data
acc_train = model.score(x_train_bow, y_train)

print('Training Accuracy:', acc_train)
acc_test = model.score(x_test_bow, y_test)
print('Test Accuracy:', acc_test)

def predict(model, vectorizer, review):
    review = clean_review(review)
    review_bow = vectorizer.transform([review])
    return model.predict(review_bow)[0]

review = 'U.S. Republican presidential candidate Donald Trump delivers a campaign speech about national security in Manchester, New Hampshire, U.S. June 13, 2016.'
predict(model, vectorizer, review)