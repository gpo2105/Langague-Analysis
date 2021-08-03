import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import nltk
import string
import re
from nltk import word_tokenize, FreqDist,regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import Processing
coverage=[str(yr) for yr in range(2011,2021,1)]

parent_path='/Users/george/Desktop/Projects/Capstone/Langague-Analysis/'
image_path=parent_path+'Images/'
data_path=parent_path+'Data/'
desc_path=os.path.join(data_path,'Desc/')
logs_path=os.path.join(data_path,'Logs/')
samples_path=os.path.join(data_path,'Samples/')


TOC_File_1=logs_path+'TOC_Success.txt'
TOC_File_2=logs_path+'TOC_Fail.txt'
TOC2_File_1=logs_path+'TOC2_Success.txt'
TOC2_File_2=logs_path+'TOC2_Fail.txt'
ID_File_1=logs_path+'ID_Success.txt'
ID_File_2=logs_path+'ID_Fail.txt'
Text_File_1=logs_path+'Text_Success.txt'
Text_File_2=logs_path+'Text_Fail.txt'
Text2_File_1=logs_path+'Text2_Success.txt'
Text2_File_2=logs_path+'Text2_Fail.txt'


log_paths={'Text':(Text_File_1,Text_File_2),
           'Text2':(Text2_File_1,Text2_File_2),
           'TOC':(TOC_File_1,TOC_File_2),
           'TOC2':(TOC2_File_1,TOC2_File_2),
           'ID':(ID_File_1,ID_File_2),
          }


Paths=pd.read_excel(logs_path+'Pathways.xlsx',
                    index_col='Ticker'
                   )

def collect_texts(ticks=None,years=None):
    flag_t=bool(ticks is not None)
    flag_y=bool(years is not None)
    if(flag_t==False and flag_y==False):
        corpus=collect_texts_all()
    elif(flag_y==False):
        if(ticks is list):
            corpus=collect_texts_stocks(ticks)
        else:
            corpus=collect_texts_stock(ticks)
    elif(flag_y==False):
        if(years is list):
            corpus=collect_texts_years(years)
        else:
            corpus=collect_texts_year(years)
    else:
        corpus={}
        for tick in ticks:
            for yr in years:
                path=Paths.loc[tick][yr]
                if(path):
                    corpus[tick+'_'+yr]=collect_text(tick,yr)
    return corpus
def collect_texts_all():
    corpus={}
    for co in Paths.index:
        for yr in Paths.columns:
            path=Paths.loc[co][yr]
            if(path):
                with open(path,'r') as f: 
                    corpus[co+'_'+yr]=f.read()
    return corpus

def collect_texts_stocks(tickers):
    corpus={}
    for tick in tickers:
        for yr,path in Paths.loc[tick].iteritems():
            if(path):
                corpus[tick+'_'+yr]=collect_text(tick,yr)
    return corpus

def collect_texts_years(years):
    corpus={}
    for yr in years:
        Y=Paths[Yr]
        for tick,path in Y.iteritems():
            if(path):
                corpus[tick+'_'+yr]=collect_text(tick,yr)

    return corpus

def collect_texts_stock(ticker):
    corpus={}
    for yr,path in Paths.loc[ticker].iteritems():
        if(path):
            corpus[yr]=collect_text(ticker,yr)
    return corpus

def collect_texts_year(year):
    corpus={}
    for tick,path in Paths[year].iteritems():
        if(path):
            corpus[tick]=collect_text(tick,year)
    return corpus
def collect_text(company,year):
    path=Paths.loc[company][year]
    with open(path,'r') as f:
        text=f.read()
        pass
    new_text=processing_text(text)
    return new_text

def processing_text(text):
    lemma=WordNetLemmatizer()
    tokens=word_tokenize(text)
    tokens=[t.lower() for t in tokens if t.lower() not in stops]
    tokens=[lemma.lemmatize(t) for t in tokens if t not in stops]
    tokens=[t for t in tokens if t not in stops]
    return ' '.join(tokens)
    
common_vocab=['result','company','may','could','risk','financial','common','issue','class',
              'stock','ability','significant','future',
              'adversly','affect','effect','impact','subject','change','continue',
              'quarter','annual','year','affect','effect','would','could','may','adverse',
              'existing','number','we','us','our',
              'including','certain','related','significant','year','ended'
             ]
             
stop_words=stopwords.words('english')
punctuations=list(string.punctuation)+["'",'\\n','``',"''"]
stops=stop_words+punctuations+common_vocab

