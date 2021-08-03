import os
import pandas as pd

__NEW__=False

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

def collected_flag(ticker,yr):
    path=samples_path+ticker+'/'+str(yr)+'.txt'
    return os.path.exists(path)



if __NEW__:
    TMTs.CIK=TMTs.apply(lambda c:c.CIK if pd.notna(c.CIK) else get_CIK(c.Ticker),axis=1)
    TMTs.to_excel(data_path+'TMT List.xlsx')
    pass
else:
    pass

if __NEW__:
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
else:
    #TMTs=pd.read_excel(data_path+'TMT List.xlsx',
    #                   converters={'CIK':str},
    #                   usecols=['Name','Ticker','CIK']
    #                  )

    #Meta=pd.read_excel(data_path+'TMT Universe Data.xlsx',
    #                       converters={'CIK':str},
    #                       index_col='Ticker'
    #                      )
    pass

if __NEW__:
    Filed=pd.DataFrame(columns=coverage,index=Meta.index)

    Filed['Filed']=Filed.index.map(lambda t:os.path.exists(samples_path+'/'+t))
    Filed['CIK']=Meta['CIK']

    for tick in Filed.index:
        for yr in years:
            Filed[str(yr)][tick]=collected_flag(tick,str(yr))
    Filed.to_excel(logs_path+'Collected.xlsx')
else:
    #Filed=pd.read_excel(logs_path+'Collected.xlsx')
    pass
if __NEW__:
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
else:
    #Paths=pd.read_excel(logs_path+'Pathways.xlsx')
    pass

