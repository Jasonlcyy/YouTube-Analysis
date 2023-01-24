#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 00:05:10 2023

@author: jasonlcyy
"""

import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import gensim
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint
import keras
import emoji

nltk.download(['stopwords', 'wordnet', 'omw-1.4', 'words'])

def clean_text(data):
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)
    
    data = emoji.demojize(str(data), delimiters=(""," "))
    
    data = re.sub('_', ' ', data)
    
    data = re.sub('[0-9]+:[0-9]+', '', data)
    
    data = re.sub('@Keo Tsang', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)
    
    data = re.sub("[^0-9a-zA-Z\s]", '', data)
    
    data = data.lower()
    
    return data

def sentences_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def words_cleaning(word_list):
    lemmatizer = WordNetLemmatizer()
    word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in stopwords.words('english') and word.isalpha()]
    return word_list
            
        
def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)


training1 = pd.read_csv("/Users/jasonlcyy/Downloads/Tweets.csv")

training2 = pd.read_csv("/Users/jasonlcyy/Downloads/Reddit_Data.csv")

training3 = pd.read_csv("/Users/jasonlcyy/Downloads/comment_training.csv")

sentiment_dict = {-1: 'negative', 0: 'neutral', 1: 'positive'}

training2 = training2.rename(columns = {'clean_comment': 'selected_text', 'category': 'sentiment'})

training2['sentiment'] = training2['sentiment'].apply(lambda x: sentiment_dict[x])

training3 = training3[['Comment Content', 'Sentiment']]

training3 = training3.rename(columns = {'Comment Content': 'selected_text', 'Sentiment': 'sentiment'})

training3['sentiment'] = training3['sentiment'].apply(lambda x: str(x).lower())

training3.dropna(inplace = True)

training1 = training1[['selected_text', 'sentiment']]

training = pd.concat([training1, training2, training3])

training['selected_text'].fillna('No Content', inplace = True)

temp_data = [clean_text(text) for text in training['selected_text'].values.tolist()]

words_data = list(sentences_to_words(temp_data))

for i in words_data:
    i = words_cleaning(i)

clean_data = [detokenize(text) for text in words_data]

sentiments = []

for label in training['sentiment'].values.tolist():
    if label == 'neutral':
        sentiments.append(0)
    elif label == 'negative':
        sentiments.append(1)
    elif label == 'positive':
        sentiments.append(2)

clean_data = np.array(clean_data)
sentiments = np.array(sentiments)

labels = tf.keras.utils.to_categorical(sentiments, 3, dtype = "float32")

max_words = 6000
max_len = 300

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(words_data)
sequences = tokenizer.texts_to_sequences(words_data)
tweets = pad_sequences(sequences, maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)

opt = Adam(lr = 0.0025)
model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(32,dropout=0.3)))
model2.add(Dense(32,activation='relu'))
model2.add(Dropout(0.3))
model2.add(layers.Dense(3,activation='softmax'))
model2.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model2.fit(X_train, y_train, epochs=30,validation_data=(X_test, y_test),callbacks=[checkpoint2])

best_model = keras.models.load_model("best_model2.hdf5")
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ',test_acc)
