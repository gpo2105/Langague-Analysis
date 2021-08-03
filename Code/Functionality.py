import os
###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as graphx
import wordcloud
####
import nltk
import string
import re
from nltk import word_tokenize, FreqDist,regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer,PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
###
from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation
from sklearn.cluster import DBSCAN, OPTICS,MeanShift
from scipy.cluster.hierarchy import dendrogram,linkage
###
from bs4 import BeautifulSoup as bs
from itertools import takewhile
import pandas_datareader
import requests
import json

######
headers=requests.utils.default_headers()
headers.update({'user-agent':'george.p.ogden@gmail.com'})
coverage=[str(yr) for yr in range(2011,2021,1)]

#######
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


#######
Universe=pd.read_excel(data_path+'TMT Universe Data.xlsx',
                       index_col='Ticker',
                       usecols=['Ticker','CIK','SIC'],
                       converters={'CIK':str}
                      )

Filed=pd.read_excel(logs_path+'Collected.xlsx',
                    index_col='Ticker',
                    converters={'CIK':str}
                   )

Paths=pd.read_excel(logs_path+'Pathways.xlsx',
                    index_col='Ticker'
                   )
######
def collect_market_data(tickers):
    START='2010-12-31'
    END='2020-12-31'
    BENCHMARK='^GSPC'
    yearly_grp=pd.Grouper(freq='A')
    #stock_prices=dict.fromkeys(tickers)
    data=pandas_datareader.DataReader(tickers,start=START,end=END,data_source='yahoo')
    daily_prices=data['Adj Close']
    #for s in stock_prices.keys():
        #data=pandas_datareader.DataReader(s,start=START,end=END,data_source='yahoo')
        #stock_prices[s]=data['Adj Close']
    #df_prices=pd.DataFrame.from_dict(stock_prices)
    df_changes=(daily_prices/daily_prices.shift(1))-1
    df_changes=df_changes.drop(pd.to_datetime(START))
    rel_changes=vs_benchmark(df_changes,BENCHMARK,START,END)
    corr=df_changes.corr()
    rel_corr=rel_changes.corr()
    annual_corr=df_changes.groupby(yearly_grp).corr()
    annual_rel_corr=rel_changes.groupby(yearly_grp).corr()
    return daily_prices,[df_changes,corr,annual_corr],[rel_changes,rel_corr,annual_rel_corr]

def vs_benchmark(daily_changes,bmrk_id,start_date,end_date):
    bmark=pandas_datareader.DataReader(bmrk_id, start=start_date,end=end_date,data_source='yahoo')['Adj Close']
    bmark_changes=(bmark/bmark.shift(1))-1
    bmark_changes=bmark_changes.drop(pd.to_datetime(start_date))
    return daily_changes.subtract(bmark_changes,axis=0)

def filter_decile(df,limit=0.9):
    means=df.abs().mean()
    idx=means[means>means.quantile(limit)].index
    return df.loc[idx,idx]
######
def collect_company_RDs_(ticker,cik,methods=False,override=False):
    print(ticker,end=': ')
    folder_path=os.path.join(sample_path,ticker)
    if(override):
        try:
            os.mkdir(folder_path)
        except OSError as error:
            print('Skip')
            return None
    else:
        pass
    files=get_10Ks_names(cik)
    for i in range(1,1+min(10,len(files))):
        yr=str(2021-i)
        file_path=folder_path+'/'+yr+'.txt'
        print(yr,end='...')
        done=False
        for method in methods:
            if(done):
                continue
            else:
                method_path=log_paths[method]
                try:
                    txt=get_Risk_Disc(cik,files[-i],method)
                    with open(file_path,'w') as f: f.write(txt)
                    with open(method_path[0],'a') as f: f.write(ticker+'--'+yr+'\n')
                    done=True
                except:
                    with open(method_path[1],'a') as f: f.write(ticker+'--'+yr+'\n')
                    continue
    print('Complete')
    

def get_CIK(ticker):
    print(ticker)
    try:
        l='https://sec.report/Ticker/'+ticker
        soup=bs(requests.get(l,headers=headers).content,features='lxml')
        return soup.find('h2').text.split()[-1]
    except:
        print('ISSUE')
        return np.nan
def get_filer_info(cik):
    #print(cik)
    l='https://data.sec.gov/submissions/CIK'+str(cik)+'.json'
    txt=requests.get(l,headers=headers).content
    meta=json.loads(txt)
    return meta['sic'],meta['fiscalYearEnd']

def get_10Ks_names(cik):
    cik_check=Universe.CIK.isin([cik])
    if(any(cik_check)):
        info=Universe[cik_check]
        file_yrs=info[coverage].dropna(axis='columns')
        return file_yrs.values.tolist()[0]
    else:
        return get_new_10Ks_names(cik)
        
def get_new_10Ks_names(cik,stop=10):
    #print(cik)
    back=pd.to_datetime('2010-12-31')
    l='https://data.sec.gov/submissions/CIK'+str(cik)+'.json'
    txt=requests.get(l,headers=headers).content
    meta=json.loads(txt)
    filings=meta['filings']
    ks=filter_10Ks(filings['recent'])
    addtnl_list=filings.get('files',[None])
    if(addtnl_list):
        addtnl_list=addtnl_list[0]
        if(pd.to_datetime(addtnl_list.get('filingTo'))>back):
            l='https://data.sec.gov/submissions/'+addtnl_list['name']
            txt=requests.get(l,headers=headers).content
            ks+=filter_10Ks(json.loads(txt))
    return ks[:min(stop,len(ks))]

def filter_10Ks(files):
    zp=zip(files['accessionNumber'],files['form'])
    return [z[0] for z in zp if z[1]=='10-K' ]


def get_filing_soup(cik,file_no):
    _,yr,file=file_no.split('-')
    link='/'.join([
                'https://www.sec.gov/Archives/edgar/data',
                cik,
                _+yr+file,
                file_no
                ])
    r=requests.get(link+'.txt',headers=headers)
    return bs(r.content,features='lxml')



def get_Risk_Disc(cik,file_no,method=None):
    min_length=100
    soup=get_filing_soup(cik,file_no)
    if(method=='Text'):
        txt=parse_RD_text(soup)
        if(min_length>len(txt)):
            raise ValueError
        else:
            return txt
    elif(method=='Text2'):
        txt=parse_RD_text_II(soup)
        if(min_length>len(txt)):
            raise ValueError
        else:
            return txt
    elif(method=='ID'):
        txt=parse_RD_ID(soup)
        if(min_length>len(txt)):
            raise ValueError
        else:
            return txt
    elif(method=='TOC'):
        txt=parse_RD_TOC(soup)
        if(min_length>len(txt)):
            raise ValueError
        else:
            return txt
    elif(method=='TOC2'):
        txt=parse_RD_TOC_II(soup)
        if(min_length>len(txt)):
            raise ValueError
        else:
            return txt
    else:
        start=soup.find_all('p',attrs={'id':'ITEM_1A_RISK_FACTORS'})[0]
        end=soup.find_all('p',attrs={'id':'ITEM_1B_UNRESOLVED_STAFF_COMMENTS'})[0]
        return parse_RD(soup,start,end)
    
    
def parse_RD(soup,start,end):
    text=''
    for x in start.find_all_next():
        if(x==end):
            break
        else:
            if(x):
                txt=x.string if(x.string) else '\n'
                text+=txt
            else:
                pass
            pass
        pass
    return text


def parse_RD_text(soup):
    S=soup.new_tag('Risk Section')
    head=soup.find(text=re.compile(r'^ITEM[\W]+1A',re.I))
    tail=head.find_next(text=re.compile(r'1B'))
    for t in takewhile(lambda tag:tag.text!=tail,head.find_all_next()):
        S.append(t)
        pass
    TOC=lambda t:bool(all([t!='Table of Contents',t!='',len(t)>2]))
    return ' \n '.join(filter(TOC,S.stripped_strings))

def parse_RD_text_II(soup):
    S=soup.new_tag('Risk Section')
    head=soup.find_all(text=re.compile(r'^ITEM[\W]+1A',re.I))[-1]
    tail=head.find_next(text=re.compile(r'1B',re.I))
    tail=tail if tail else head.find_next(text=re.compile(r'^ITEM',re.I))
    for t in takewhile(lambda tag:tag.text!=tail,head.find_all_next()):
        S.append(t)
        pass
    TOC=lambda t:bool(all([t!='Table of Contents',t!='',len(t)>2]))
    return ' \n '.join(filter(TOC,S.stripped_strings))

def parse_RD_ID(soup):
    starter=soup.find(['a','p'],id=re.compile(r'Risk_?Factor',re.I))
    if(starter):
        stopper=soup.find(['a','p'],id=re.compile(r'UnResolved',re.I))
    else:
        starter=soup.find(['a','p'],{'name':re.compile(r'Risk_?Factor',re.I)})
        stopper=soup.find(['a','p'],{'name':re.compile(r'UnResolved',re.I)})
    first_p=starter.find_next(['p'])
    last_p=stopper.find_next('p')
    S=soup.new_tag('Risk Section')
    first_p.insert_before(S)
    for tag in takewhile(lambda t:t!=last_p,S.find_all_next('p')):
        S.append(tag)
    cleaned=filter(lambda txt:len(txt)>2,S.stripped_strings)
    L=[text for text in cleaned if text!='Table of Contents']
    #start=starter.find_next(['p','div'])
    #stop=stopper.find_next(['p','p'])
    return ' \n '.join(L)


def parse_RD_TOC(soup):
    start_id=soup.find(['a'],text=re.compile(r'(1A|Risk)'))['href'][1:]
    stop_id=soup.find(['a'],text=re.compile(r'1B|Unresolved'))['href'][1:]
    start,stop=bind_RD(soup,start_id,stop_id)
    return iterate_tags(start,stop)

def parse_RD_TOC_II(soup):
    #text=r''
    start_row=soup.find(['td'],text=re.compile(r'(1A|Risk)'))
    start_id=start_row.find_next('td').find_next('a')['href'][1:]
    stop_row=soup.find(['td'],text=re.compile(r'(1B|Unresolved)'))
    stop_id=stop_row.find_next('td').find_next('a')['href'][1:]
    start,stop=bind_RD(soup,start_id,stop_id)
    return iterate_tags(start,stop)
    #Iter=takewhile(lambda t:t.a!=stop,start.find_all_next())
    #for t in Iter:
        #s=' '.join(t.stripped_strings)
        #text+=' \n '+s
    #return text
def bind_RD(soup,start_id,stop_id):
    if(start_id==None or stop_id==None):
        raise ValueExcetpion
    else:
        start=soup.find(['a','div','span'],id=start_id)
        start=start if start else soup.find(['a','div','span'],{'name':start_id})
        stop=soup.find(['a','div','span'],id=stop_id)
        stop=stop if stop else soup.find(['a','div','span'],{'name':stop_id})
        return (start,stop)
def iterate_tags(start,stop):
    strings=[]
    tags=start.next.find_next_siblings()
    for t in tags:
        if(t==stop or t.a==stop):
            #print(t)
            break
        else:
            strings.append(' '.join(t.stripped_strings))
    #        print(' '.join(t.stripped_strings))
            pass
    TOC=lambda t:bool(all([t!='Table of Contents',t!='',len(t)>2]))
    cleaned=[s for s in filter(TOC,strings)]
    return r' \n '.join(cleaned)



def get_sic_des(label):
    link='https://www.osha.gov/sic-manual/{section}'
    r=requests.get(link.format(section=label))
    soup=bs(r.content,'lxml')
    content=soup.find('div',attrs={'id':'main-content'}).find_next('div')
    des='  '.join(content.stripped_strings)
    return des

def get_co_desc(ticker):
    path=desc_path+'Cos/'+ticker+'.txt'
    l='https://finance.yahoo.com/quote/'+ticker+'/profile'
    soup=bs(requests.get(l,headers=headers).content,'lxml')
    sector=soup.find('span',text=re.compile('Sector')).find_next('span').text
    industry=soup.find('span',text=re.compile('Industry')).find_next('span').text
    des=soup.find('h2',text='Description').find_next('p').text
    with open(path,'w') as f:
        f.writelines([sector,'\n',industry,'\n',des])
    return None

def mannual_extract(ticker,yr):
    base='/Users/george/Desktop/Projects/Capstone/Langague-Analysis/Data/'
    man_path=base+'Logs/Manual.txt'
    path=base+'Samples/'+ticker+'/'+yr+'.txt'
    with open(path,'r') as f:
        raw=f.read()
    soup=bs(raw,'lxml')
    text=r''
    for s in soup.stripped_strings:
        txt=s if (len(s)>2 and re.match(r'Table of',s)==None) else ''
        text+=r' \n '+txt
    with open(path,'w') as f:  f.write(text)
    with open(man_path,'a') as f: f.write(ticker+'--'+yr+'\n')
    return text
#######

def collected_flag(ticker,yr):
    path=samples_path+ticker+'/'+str(yr)+'.txt'
    return os.path.exists(path)

def update():
    TMTs.CIK=TMTs.apply(lambda c:c.CIK if pd.notna(c.CIK) else get_CIK(c.Ticker),axis=1)
    TMTs.to_excel(data_path+'TMT List.xlsx')

    Meta=TMTs.CIK.apply(get_filer_info)
    Meta=pd.DataFrame(Meta.to_list(),columns=['SIC','FY'])
    Meta=pd.concat([TMTs,Meta],axis=1)
    filings=pd.DataFrame(Meta['CIK'].apply(get_10Ks_names).to_list())
    filings.columns=[str(yr) for yr in range(2020,2010,-1)]
    Meta=pd.concat([Meta,filings],axis=1)
    Meta=Meta.set_index(['Ticker'])
    Meta.to_excel(data_path+'TMT Universe Data.xlsx')
    for tick in TMTs.Ticker:
        if(os.path.exists(desc_path+'Co/'+tick+'.txt')):
            pass
        else:
            get_co_desc(tick)

    for sic_code in Meta.SIC.value_counts().index:
        print(sic_code)
        txt=get_sic_des(sic_code)
        with open(desc_path+'SICs/'+str(sic_code)+'.txt','w') as f:
            f.write(txt)

    Filed=pd.DataFrame(columns=coverage,index=Meta.index)

    Filed['Filed']=Filed.index.map(lambda t:os.path.exists(samples_path+'/'+t))
    Filed['CIK']=Meta['CIK']

    for tick in Filed.index:
        for yr in years:
            Filed[str(yr)][tick]=collected_flag(tick,str(yr))
    Filed.to_excel(logs_path+'Collected.xlsx')

    Paths=pd.DataFrame(index=Filed.index,columns=coverage)
    for co in Paths.index:
        for y in Paths.columns:
            new_path=samples_path+co+'/'+y+'.txt'
            if(os.path.isfile(new_path)):
                Paths.loc[co][y]=new_path
            else:
                Paths.loc[co][y]=False
                pass
            pass
        pass
    Paths.to_excel(logs_path+'Pathways.xlsx')



#######
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


######

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
    RDs=collect_texts_stock(ticker)
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



#####

def create_extractor(texts,vector_args):
    tf=TFidVectorizer(stop_words=stops,*vector_args)
    tf.fit(texts)
    return tf