#!/usr/bin/env python
# coding: utf-8

# In[7]:


# importing packages
import pandas as pd
import numpy as np
import re
from nltk import RegexpParser
from gensim import models
import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import heapq
from operator import itemgetter
import nltk
import gensim
import string
import random
from nltk.corpus import subjectivity,stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.corpus import stopwords 
from nltk import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
import logging
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases


# In[49]:



def topicmodel_lda(nlimit, infile_path, outfile_path, col_name):
    """
    Give the topics of the text provided in the column.

    Parameters:
    ----------
    column : Column in the dataframe which contains the text data for which topic modelling has to be done after removing the NAs.
    nlimit: Maximum number of topics.
    
    Returns:
    -------
    Topics
    """
    df = pd.read_excel(infile_path)
    column = df[col_name]
    
    stop_words_final = {'l', "should've", "mustn't", 'themselves', 'by', 'up', 'such', 'oh', 'very', 'full', 'shops', "doesn't", 'those', 'youd', 'yourselves', 'today', "you'd", 'into', 'am', 'ain', 'phone', 'between', 'feel', 'several', 'if', 'd', 'offered', 'don', 'souq', 'having', 'off', 'again', 'which', 'z', "hadn't", 'u', 'auto', 'im', 'every', 'who', 'one', 'll', "wouldn't", 'verified', 'were', 'month', 'page', 'br', 'product', 'while', 'needn', 'feels', 'the', 'thanks', 'call', 'x', 'color', 'do', 'until', 'so', 'too', 'that', 'honor', 'make', 'com', 'above', 'all', 'this', "aren't", 'even', 'paper', 'where', 'and', 'youre', 'v', 'me', 'ourselves', 'g', 'shan', 'shipping', 'please', 'LOL', 'etc', 'centre', 'but', 't', 'article', 'jumia', 'isn', 'much', 'p', 'an', 'r', "don't", 'you', 'through', 'when', 'does', 'year', 'against', 'doing', 'than', 'then', 'be', "won't", 'couldn', 'more', 'of', 'because', "you've", 'after', 'some', 'down', 'offer', 'in', 'now', 'h', 'her', 'n', 'theirs', 'anymore', "shouldn't", 'i', 'haven', "you'll", 'youve', 'articles', 'thats', 'been', 'ps', 'most', 'oppo', 'k', 'ma', 'lol', 'able', 'here', 'dont', 'him', 'should', 'ready', 'purchase', 'being', 'before', 'what', 'wouldn', 'why', 'section', 'your', 'company', 'y', 'his', 'below', 'shant', 'find', 'back', 'she', 're', 'thatll', 'or', 'at', 'any', 'new', 'think', 'few', 'recieved', 'j', 'under', 'there', 'mightn', 'f', 'b', 'towards', 'yours', 'other', 'e', 'want', 'its', 'thank', 'is', 'iphone', 'he', 'really', 'called', 'mobile', 'how', 'chosen', 'my', "she's", 'gb', "wasn't", 'w', 'take', 'their', 'myself', 'going', "it's", 'let', 'are', 'answer', "haven't", 'weren', 'need', 'question', 'with', 'have', 'pro', 'share', 'ours', 'right', "weren't", 'shouldve', 'was', 'many', "you're", 'own', 'read', 'delivered', 'latest', 'itself', 'didn', "isn't", 'questions', 'hasn', 'thing', 'sure', 'colour', 'lg', 'c', "couldn't", 'yourself', 'they', 'know', 'hadn', 'reading', 'hour', 'a', "didn't", 'news', 'address', 'style', 'whom', 'nokia', 've', 'will', 'like', 'shes', 'to', 'o', 'huawei', "that'll", 'has', 'wasn', 'times', 'yesterday', "hasn't", 'we', 'each', 'look', 'over', 'our', "shan't", 'hers', 'nor', 'did', 'told', 'part', 'had', 'won', 'could', 'shouldn', 'same', 'also', 'just', 's', 'during', 'it', 'discount', 'size', 'them', 'for', 'comment', 'further', 'youll', 'out', 'doesn', 'm', 'only', 'change', 'aren', 'must', 'from', 'can', 'on', 'go', 'both', 'require', 'using', "mightn't", 'himself', 'ask', 'todays', 'q', 'delivery', 'herself', 'would', 'once', 'ive', 'caller', 'mustn', 'about', 'these', 'based', "needn't", 'best', 'as'}
    
    column1 = column.fillna("NA")
    column1 = column1[column1!='NA']
    
    def stopword_removal(match): 
        filtered_sentence = []
        word_tokens = word_tokenize(match)
        for w in word_tokens:
            if w not in stop_words_final:
                filtered_sentence.append(w)
        return ' '.join(filtered_sentence) 

    def docs_preprocessor(docs):
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = docs[idx].lower() 
            docs[idx] = stopword_removal(docs[idx])
            docs[idx] = tokenizer.tokenize(docs[idx])  

        return docs
    
    def text_cleanup(match):
        #match = re.sub("\w\,",'',match)
        match = re.sub(r'^\w+\,?\s','',match)
        #match = re.sub("Color: [\w\s]*\|",'',match)
        #match = re.sub("Style: [\w\s]*\|",'',match)
        #match = re.sub("Size: [\w\s]*\|",'',match)
        match = match.lower()
        #match = match.replace("no cost emi", "")
        #match = match.replace("additional exchange", "")
        match = match.replace("won't",'will not')
        match = match.replace("shouldn't",'should not')
        match = match.replace("aren't",'are not')
        match = match.replace("couldn't",'could not')
        match = match.replace("doesn't",'does not')
        match = match.replace(r"isn't", "is not")
        match = match.replace(r"weren't", "were not")
        match = match.replace(r"hasn't", "has not")
        match = match.replace(r"hadn't", "had not")
        match = ''.join([i for i in match if not i.isdigit()])
        #Remove additional white spaces
        match = re.sub('[\s]+', ' ', match)
        # Remove '.' in between a sentence
        match = re.sub('\s[\\.]\s', ' ', match)
        #remove individual characters
        shortword = re.compile(r'\W*\b\w{1}\b')
        match = shortword.sub('', match)
        match = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',match)
        #Remove additional white spaces
        match = re.sub('[\s]+', ' ', match)
        #trim
        match = match.strip('\'"')
        match = re.sub('[-=/!%@#$;():~]', '', match)
        match = re.sub('[^a-z]+',' ',match)
        match = stopword_removal(match)
        #match = lemma_fun(match)
        return match
    
    def compute_coherence_values(dictionary, corpus, texts, limit, start=1, step=1):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
#         coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,alpha='auto', passes=20, iterations=100)
            model_list.append(model)
#             coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#             coherence_values.append(coherencemodel.get_coherence())

        return model_list
    

    column1 = list(map(lambda x: text_cleanup(x), column1))

#     docs = array(column1)
    
    
    docs = docs_preprocessor(column1)
   
    
    bigram = Phrases(docs, min_count=10)
    trigram = Phrases(bigram[docs])

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]: #?????
            if '_' in token:

                docs[idx].append(token)
        for token in trigram[docs[idx]]: #?????
            if '_' in token:

                docs[idx].append(token)

    dictionary = Dictionary(docs)

    dictionary.filter_extremes(no_below=3, no_above=0.2)

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    model_list = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=1, limit=nlimit+1, step=1)
    
    
    lda_model = model_list[nlimit-1]
    list_topics = []
    index = []
    for i in range(nlimit):
        list_topics.append(lda_model.print_topics()[i][1].split('+'))
        index.append('Topic'+str(i))
    
    topic_df = pd.DataFrame(list_topics, index=index)
    topic_df.to_excel(outfile_path + '\\Topic_modeling_output.xlsx')
    

