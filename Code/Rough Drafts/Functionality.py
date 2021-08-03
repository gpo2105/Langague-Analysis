import os
import sys
import importlib
from bs4 import BeautifulSoup as bs
import urllib
import tempfile
import re
import difflib
import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import takewhile
import requests
import json

headers=requests.utils.default_headers()
headers.update({'user-agent':'george.p.ogden@gmail.com'})
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

Universe=pd.read_excel(data_path+'TMT Universe Data.xlsx',
                       index_col='Ticker',
                       usecols=['Ticker','CIK','SIC'],
                       converters={'CIK':str}
                      )

Filed=pd.read_excel(logs_path+'Collected.xlsx',
                    index_col='Ticker',
                    converters={'CIK':str}
                   )

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



def get_mkt_data(tickers,start_date,end_date):
    yearly_grp=pd.Grouper(freq='A')
    quarterly_grp=pd.Grouper(freq='Q')
    monthly_gro=pd.Grouper(freq='M')
    stock_prices=dict.fromkeys(tickers)
    for s in tickers:
        stock_prices[s]=pandas_datareader.DataReader(s, 
                       start=start_date, 
                       end=end_date, 
                       data_source='yahoo')['Adj Close']
    df_prices=pd.DataFrame.from_dict(stock_prices)
    df_changes=(df_prices/df_prices.shift(1))-1
    df_changes=df_changes.drop(start_date)
    return df_price,df_changes

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


    