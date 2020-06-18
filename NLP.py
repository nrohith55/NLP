# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:11:07 2020

@author:Rohith
"""
###############Varuns Class Basics of NLP
import numpy as np
import pandas as pd
import os
import nltk
import nltk.corpus
nltk.download('all')
AI='This video will provide you with a comprehensive and detailed knowledge of Natural Language Processing, popularly known as NLP. You will also learn about the different steps involved in processing the human language like Tokenization, Stemming, Lemmatization and more. Python, NLTK, & Jupyter Notebook are used to demonstrate the concepts'

type(AI)

from nltk.tokenize import word_tokenize# To create tokens

AI_token=word_tokenize(AI)
AI_token

len(AI_token)#To find the length of tokens

from nltk.probability import FreqDist#To check how many times tokens are repeated

fdist=FreqDist()

for word in AI_token:
    fdist[word.lower()]+=1
fdist

fdist['the']

len(fdist)

fdist_top20=fdist.most_common(5)
fdist_top20

from nltk.tokenize import blankline_tokenize
AI_blankline=blankline_tokenize(AI)
AI_blankline
AI_blankline[0]

from nltk.util import bigrams,trigrams,ngrams ####to use bigram,trigram and ngram
AI_bigrams=list(bigrams(AI_token))
AI_bigrams

from nltk.stem import PorterStemmer#Steps to do stemming

stemmer=PorterStemmer()

stemmer.stem('concepts')
PorterStemmer().stem('Jumping')

#NOTE :We can use Porterstemmer or SnowballStemmer

from nltk.stem import wordnet#Steps to do Lemmatization
from nltk.stem import WordNetLemmatizer

y=WordNetLemmatizer()

y.lemmatize('corpora')

from nltk.corpus import stopwords

stopwords.words('english')

########Byom Class :Basics of NLP


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


input_string_1 = "This course on AI is a good course, where we will try to cover multiple concepts."
input_string_2 = "This course on AI is a Good Course, where We will try to cover multiple Concepts."

input_string_1.lower()

input_string_2.lower()

print(input_string_1==input_string_2)

similarity_between_sentences=input_string_1.lower()==input_string_2.lower()

print(similarity_between_sentences)

###################################################

spel_1 = "i like avenger but not a Great Movie for grown-ups"
spel_2 = "i likes  avenger but not a great movie for grown ups"

spel_similarity=spel_1==spel_2

print(spel_similarity)

##################################################

spel_1=spel_1.lower().replace("  "," ").replace("-"," ")
print(spel_1)
spel_2=spel_2.lower().replace("  "," ")
print(spel_2)

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
stemming=ps.stem("likes")
print(stemming)


lem=WordNetLemmatizer()
lemmatization=lem.lemmatize('likes')
print(lemmatization)

###################################################
#Creating for loop to apply stemming .lemmatization

from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()


def apply_stem(sentence):
    
    clean_list=[]
    sentence=sentence.split()
    for word in sentence:
        word=lem.lemmatize(word)
        clean_list.append(word)
        
    clean_sentence=" ".join(clean_list)
    
    return clean_sentence

spel_1 = "i like avenger but not a Great Movie for grown-ups"
spel_2 = "i likes  avenger but not a great movie for grown ups"

clean_spel_1 = apply_stem(spel_1)
clean_spel_2 = apply_stem(spel_2)

print (clean_spel_1)
print (clean_spel_2)

###################################################################

#Creating loop for removing stopwords

from nltk.corpus import stopwords

def remove_stopwords(sentence):
    stop_words=set(stopwords.words('english'))
    modified_stop_words=list(stop_words)
    
    clean_words=[]
    sentence=sentence.split()
    for word in sentence:
        if word.lower() in set( modified_stop_words):
            pass
        else:
            clean_words.append(word)
            
    clean_sentence=" ".join(clean_words)
    
    return clean_sentence

ugc_1 = "This mobile is good, but display is not good"
ugc_2 = "This mobile good, but display is not good"

ugc_clean_1 = remove_stopwords(ugc_1)
ugc_clean_2 = remove_stopwords(ugc_2)

print (ugc_clean_1)
print (ugc_clean_2)



###################################################################################
    
            
    
    























        












































