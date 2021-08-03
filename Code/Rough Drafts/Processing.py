from nltk import word_tokenize, FreqDist,regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re

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

def process_text(text):
    lemma=WordNetLemmatizer()
    tokens=word_tokenize(text)
    tokens=[t.lower() for t in tokens if t.lower() not in stops]
    tokens=[lemma.lemmatize(t) for t in tokens if t not in stops]
    tokens=[t for t in tokens if t not in stops]
    return ' '.join(tokens)