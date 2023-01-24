#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:10:43 2023

@author: jasonlcyy
"""
import pandas as pd
import re
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim
import emoji
from nltk.corpus import stopwords

comments_1 = pd.read_csv("/Users/jasonlcyy/Downloads/comments (2).csv")

comments_2 = pd.read_csv("/Users/jasonlcyy/Downloads/comments (3).csv")

comments_3 = pd.read_csv("/Users/jasonlcyy/Downloads/comments (4).csv")

comments_4 = pd.read_csv("/Users/jasonlcyy/Downloads/comments (5).csv")

comments_5 = pd.read_csv("/Users/jasonlcyy/Downloads/comments (7).csv")

def clean_text(data):  
    data = emoji.demojize(str(data), delimiters=(""," "))
    
    data = re.sub('_', ' ', data)
    
    data = re.sub('[0-9]+:[0-9]+', '', data)
    
    data = re.sub('@Keo Tsang', '', data)
    
    data = re.sub('@[A-Za-z]+', '', data)
    
    data = re.sub("[^0-9a-zA-Z\s\!\.\,\?]", '', data)
    
    # Remove new line characters
    data = re.sub('\s+', ' ', data)
    
    data = data.lower()
    
    return data

def clean_wordcloud(data):
    data = emoji.demojize(str(data), delimiters=(""," "))
    
    data = re.sub('_', ' ', data)
    
    data = re.sub('[0-9]+:[0-9]+', '', data)
    
    data = re.sub("[^0-9a-zA-Z\s\!\.\,\?]", '', data)
    
    # Remove new line characters
    data = re.sub('\s+', ' ', data)
    
    return data
    

def sentiment_analyse(comment_list):
    sequence = tokenizer.texts_to_sequences(comment_list)
    test = pad_sequences(sequence, maxlen=300)
    sentiment[np.around(sentiment_analyzer.predict(test), decimals=0).argmax(axis=1)[0]]

def sentences_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
comments = pd.concat([comments_1, comments_2, comments_3, comments_4, comments_5])

comments.drop(['Comment Author URL', 'Email Addresses', 'Is Thread', 
               'Comment Sentiment Score', 'Comment URL', 'Asset Author ID',
               'Asset Author Name', 'Asset Author URL'], axis = 1, inplace = True)

comments['Comment Published Date'] = pd.to_datetime(comments[
    'Comment Published Date'])
comments['Asset Published Date'] = pd.to_datetime(comments[
    'Asset Published Date'])


''' word cloud '''
word_cloud_list = comments[['Comment Content']]

word_cloud_list['Comment Content'] = word_cloud_list['Comment Content'].apply(
    lambda x: clean_wordcloud(x))

word_cloud_sentence = word_cloud_list['Comment Content'].tolist()

word_list = list(sentences_to_words(word_cloud_sentence))

stop_words = set(stopwords.words('english'))

word_cloud = [y for x in word_list for y in x if not y.lower() in stop_words]

word_cloud = pd.DataFrame(data = word_cloud, columns = ['word'])

word_cloud.to_csv('/Users/jasonlcyy/Downloads/word_cloud.csv')

'''sentiment analysis'''
clean_comments = comments[['Comment Content']]

clean_comments['Comment Content'] = clean_comments['Comment Content'].apply(
    lambda x: clean_text(x))

sentence_list = clean_comments['Comment Content'].tolist()

sentiment = ['Neutral','Negative','Positive']

tokenizer = Tokenizer(num_words=6000)

tokenizer.fit_on_texts(sentence_list)

sentiment_analyzer = load_model('/Users/jasonlcyy/Desktop/best_model2.hdf5')

comment_sentiment = []
sentiment_score = []

for sentence in sentence_list:
    sequence = tokenizer.texts_to_sequences([sentence])
    print(sequence)
    test = pad_sequences(sequence, maxlen=300)
    print(sentence)
    prediction = sentiment_analyzer.predict(test)
    print(sentiment[np.around(prediction, decimals=0).argmax(axis=1)[0]])
    comment_sentiment.append(sentiment[np.around(
        prediction, decimals=0).argmax(axis=1)[0]])
    sentiment_score.append(np.max(prediction))
    
clean_comments['sentiment'] = comment_sentiment
clean_comments['Confidence'] = sentiment_score

clean = clean_comments.drop_duplicates()

sentiment_export = pd.merge(comments, clean, how = 'left', on = 'Comment Content')

sentiment_export.to_csv('/Users/jasonlcyy/Downloads/sentiment_comments.csv')

''' video metadata '''
video_metadata = comments[['Asset Title', 
                           'Asset Published Date', 
                           'Asset Views Count', 
                           'Asset Likes Count',
                           'Asset URL']].drop_duplicates()

comment_count = comments[
    comments['Comment Author Name']!='Keo Tsang'].groupby(
        'Asset Title').size().reset_index()
        
video_metadata = video_metadata.merge(comment_count, on='Asset Title')

del comment_count

video_metadata.rename(columns={'Asset Title': 'Video Title',
                               'Asset Views Count': 'Views',
                               'Asset Published Date': 'Date',
                               'Asset Likes Count': 'Likes',
                               0: 'Number of Comments'},
                               inplace = True)

video_metadata['comments-to-views ratio'] = video_metadata[
    'Number of Comments']/video_metadata['Views']
video_metadata['likes-to-views ratio'] = video_metadata[
    'Likes']/video_metadata['Views']

video_metadata.sort_values(by = 'Date', ascending = False, inplace = True)

video_metadata.reset_index(drop=True, inplace=True)

video_metadata.to_csv('/Users/jasonlcyy/Downloads/video_metadata.csv')

''' most loyal commenter and their comments '''

most_loyal = comments.groupby(by =[
    'Comment Author ID', 'Comment Author Name']).count().reset_index()[[
        'Comment Author ID', 'Comment Author Name', 'Comment Likes Count']]

most_loyal.rename(columns={
    'Comment Likes Count': 'Number of Comments'}, inplace = True)
most_loyal = most_loyal.sort_values(
    by = 'Number of Comments', ascending = False).drop(
    ['Comment Author ID'], axis=1).head(20)

most_loyal_comments = comments[
    comments['Comment Author Name']==most_loyal[
        'Comment Author Name'].iloc[1]][[
    'Asset Title', 'Comment Author Name', 'Comment Content']]

for author in most_loyal['Comment Author Name'].iloc[2:11].tolist():
    temp_df = comments[comments['Comment Author Name']==author][[
        'Asset Title', 'Comment Author Name', 'Comment Content']]
    most_loyal_comments = pd.concat([most_loyal_comments, temp_df])

''' fastest comments '''
comments['Speed']=comments[
    'Comment Published Date']-comments['Asset Published Date']
fastest = comments.sort_values(by = 'Speed', ascending = True).groupby(
    by = 'Asset Title').head(10)
fastest.sort_values(by = [
    'Asset Published Date', 'Comment Published Date'], inplace = True)
fastest.drop(['Comment ID', 'Comment Author ID', 
              'Asset ID', 'Asset Views Count',
            'Asset Likes Count', 
            'Asset Dislikes Count', 'Asset Comments Count'],
             axis = 1, inplace = True)

fastest_commenters = fastest.groupby(
    by = 'Comment Author Name').count().sort_values(
        by = 'Speed', ascending = False)['Speed']

# Probably notification switched on
fastest_commenters = fastest.groupby(
    by = 'Comment Author Name').count().sort_values(
        by = 'Speed', ascending = False)['Speed']
        
fastest_commenters.to_csv('/Users/jasonlcyy/Downloads/fastest_commenters.csv')

''' most liked/replied comments '''
most_liked = comments.sort_values(
    by = 'Comment Likes Count', ascending = False).groupby(
        by = 'Asset Title').head(10)
most_liked.sort_values(
    by = ['Asset Published Date', 'Comment Likes Count'], 
    ascending = [True, False], inplace = True)
most_liked.drop(['Comment ID', 'Comment Author ID', 
                 'Asset ID', 'Asset Views Count',
                 'Asset Likes Count', 'Asset Dislikes Count', 
                 'Asset Comments Count'],
             axis = 1, inplace = True)
most_replied = comments.sort_values(
    by = 'Comment Replies Count', ascending = False).groupby(
        by = 'Asset Title').head(10)
most_replied.sort_values(
    by = ['Asset Published Date', 'Comment Replies Count'], 
    ascending = [True, False], inplace = True)
most_replied.drop([
    'Comment ID', 'Comment Author ID', 
    'Asset ID', 'Asset Views Count',
    'Asset Likes Count', 'Asset Dislikes Count', 'Asset Comments Count'],
             axis = 1, inplace = True)