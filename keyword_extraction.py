'''
Experimental, under development
'''

import re
from pandas import read_csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from textblob import TextBlob, Word
import spacy
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from re import sub
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))
#w1 = read_csv('wave1_website_text_true_false.csv')

def extract_experimental(text):
    text_sans_n = re.sub('\n', ' ', text)
    sentences = sent_tokenize(text_sans_n)
    lemmatizer = WordNetLemmatizer()
    d_sentence = {}
    lst_keywords = []
    nlp = spacy.load("en_core_web_sm")

    for idx, sent in enumerate(sentences):
        print(sent)
        doc = nlp(sent)
        words = word_tokenize(sent.lower())
        lst_keywords_temp = []
        for chunk in doc.noun_chunks:
            print(str(chunk.text)+", "+str(chunk.root.text)+", "+str(chunk.root.dep_)+", "+str(chunk.root.head.text))
            print('\n\n')
        
        for i in nltk.pos_tag(words):
            if i[1]=='NN' or i[1]=='NNP' or i[1]=='NNS' or i[1]=='NNPS':
                lst_keywords_temp.append(lemmatizer.lemmatize(i[0]))
                #print(lst_keywords_temp[-1], end=' OR ')
        
        lst_keywords.append(lst_keywords_temp)
        
        print('\n\n')

