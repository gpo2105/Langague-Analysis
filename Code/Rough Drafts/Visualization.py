import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud

import nltk
from nltk import word_tokenize, FreqDist,regexp_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import re

import os

import Textual as Txt
import Processing

parent_path='/Users/george/Desktop/Projects/Capstone/Langague-Analysis/'
image_path=parent_path+'Images/'
data_path=parent_path+'Data/'
samples_path=os.path.join(data_path,'Samples/')
logs_path=os.path.join(data_path,'Logs/')



rc_map={
        1:(1,1),2:(1,2),3:(1,3),4:(1,4),5:(1,5),
        6:(2,3),7:(2,4),8:(2,4),9:(3,3),10:(2,5)
        }

def visualize_stock(ticker):
    try:
        os.mkdir(image_path+'Wordclouds/Companies/'+ticker)
    except:
        pass
    
    stock_all(ticker)
    stock_timeline(ticker)
    return None

def stock_all(ticker):
    path=image_path+'Wordclouds/Companies/'+ticker+'/'
    RDs=Txt.collect_texts_stock(ticker)
    text=' '.join(RDs.values())
    wc=wordcloud.WordCloud()
    wc.min_word_length=3
    wc.generate_from_text(text.lower())
    fig=plt.subplot()
    fig.set_title(ticker+':  All Risk Disclosures')
    fig.imshow(wc.to_array());
    plt.savefig(path+'Combined.pdf',
                orientation='landscape',
                pad_inches=0.0,
                bbox_inches='tight',
                format='pdf'
               )
    return None

def stock_timeline(ticker):
    path=image_path+'Wordclouds/Companies/'+ticker+'/'
    RDs=collect_texts_stock(ticker)
    years=list(RDs.keys())
    years.sort()
    l=len(RDs)
    if l>1:
        rows,cols=rc_map[l]
        wc=wordcloud.WordCloud()
        wc.min_word_length=3
        fig,axes=plt.subplots(ncols=cols,
                              nrows=rows,
                              figsize=(27,9),
                              tight_layout=True,
                              sharex=True,
                              sharey=True
                             )

        axes=axes.reshape(-1)
        i=0
        for yr in years:
            a=axes[i]
            a.frameon=False
            a.set_xticklabels([])
            a.set_yticklabels([])
            text=RDs[yr]
            wc.generate_from_text(text.lower())
            a.set_title(str(yr),fontsize='large')
            a.imshow(wc.to_array());
            i+=1
        plt.savefig(path+'Timeline.pdf',
                    orientation='landscape',
                    pad_inches=0.0,
                    bbox_inches='tight',
                    format='pdf'
                   )
        pass
    else:
        pass
    return None