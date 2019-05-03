# Citations:
# https://stackoverflow.com/questions/11023411/how-to-import-csv-data-file-into-scikit-learn
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


tf.logging.set_verbosity(tf.logging.INFO)

f1 = pd.read_csv("train-3.csv", header=0)

# organize data

id = f1.values[:,0]
sentences = f1.values[:,1]
words = list()

for sentence in sentences:
    for word in sentence.split():
        words.append(word)

authors = f1.values[:,2]
classes = ['EAP', 'HPL', 'MWS']


""" SKLEARN: PROJECT PT 1 """
#  vectorize data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(sentences)
#print(X_train_counts)

#  transform data using tfid
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
#print(X_train_tf)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf)

#  naive bayes
clf = MultinomialNB().fit(X_train_tfidf, authors)

X_new_counts = count_vect.transform(sentences)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(sentences, predicted):
#      print('%r => %s' % (doc, category))

# get lower score with CountVectorizer(binary=True) : default = False


#  naive bayes pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),

])

#  fit data with naive bayes pipeline
text_clf.fit(sentences, authors)

predicted = text_clf.predict(sentences)
print('MultinomialNB:', np.mean(predicted == authors))

#  svc pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classify', SVC(kernel='linear')),
])

#  fit data with svc pipeline
text_clf.fit(sentences, authors)

predicted = text_clf.predict(sentences)
print('SVM:', np.mean(predicted == authors))