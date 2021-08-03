import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
import string
import re
from nltk import word_tokenize, FreqDist,regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation
from sklearn.cluster import DBSCAN, OPTICS
from scipy.cluster.hierarchy import dendrogram,linkage

types={
        'Agg':AgglomerativeClustering,
        'KM':KMeans,
        'Aff':AffinityPropagation,
        'DBS':DBSCAN,
        'OPT':OPTICS
        }
options={
        'Agg':[],
        'KM':['N'],
        'Aff':[],
        'DBS':[],
        'OPT':[]
    }



def create_extractor(texts,vector_args):
    tf=TFidVectorizer(stop_words=stops,*vector_args)
    tf.fit(texts)
    return tf