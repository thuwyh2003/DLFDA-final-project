import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm

import os
import zipfile
from joblib import delayed, Parallel
from itertools import product
from utils import CHARAS_LIST

import warnings
warnings.filterwarnings('ignore')

if 'data' not in os.listdir():
    os.mkdir('data')
    os.system("wget https://cloud.tsinghua.edu.cn/seafhttp/files/dbe14489-0c14-407c-9745-5475e2f596d7/datashare_re.pkl -O datashare.pkl")
    os.system("wget https://cloud.tsinghua.edu.cn/seafhttp/files/5512a9bd-8747-4c82-80b6-bd2a0722e08d/mon_list.pkl -O mon_list.pkl")
    os.system("wget https://cloud.tsinghua.edu.cn/seafhttp/files/721583aa-a9a1-4246-9c23-f7c310dc75fa/month_ret.pkl -O month_ret.pkl")
    os.system('wget https://cloud.tsinghua.edu.cn/f/b93c6ae7e2014d3a951e/?dl=1 -O ff5.csv')
    os.system('wget https://cloud.tsinghua.edu.cn/f/5f077be9eda0428ab7e5/?dl=1 -O UMD.csv')
    
    os.system('mv ff5.csv data')
    os.system('mv UMD.csv data')
    os.system('mv mon_list.pkl data')
    os.system('mv datashare.pkl data')
    os.system('mv month_ret.pkl data')

    
    datashare=pd.read_pickle('data/datashare.pkl')
    mon_ret=pd.read_pickle('data/month_ret.pkl')



def pre_process(date):
    cross_slice = datashare.loc[datashare.DATE == date].copy(deep=False)
    omitted_mask = 1.0 * np.isnan(cross_slice.loc[cross_slice['DATE'] == date])
    # fill nan values with each factors median
    cross_slice.loc[cross_slice.DATE == date] = cross_slice.fillna(0) + omitted_mask * cross_slice.median()
    # if all stocks' factor is nan, fill by zero
    cross_slice.loc[cross_slice.DATE == date] = cross_slice.fillna(0)

    re_df = []
    # rank normalization
    for col in CHARAS_LIST:
        series = cross_slice[col]
        de_duplicate_slice = pd.DataFrame(series.drop_duplicates().to_list(), columns=['chara'])
        series = pd.DataFrame(series.to_list(), columns=['chara'])
        # sort and assign rank, the same value should have the same rank
        de_duplicate_slice['sort_rank'] = de_duplicate_slice['chara'].argsort().argsort()
        rank = pd.merge(series, de_duplicate_slice, left_on='chara', right_on='chara', how='right')['sort_rank']
        # if all values are zero, the results will contain nan
        rank_normal = ((rank - rank.min())/(rank.max() - rank.min())*2 - 1)
        re_df.append(rank_normal)
    re_df = pd.DataFrame(re_df, index=CHARAS_LIST).T.fillna(0)
    re_df['permno'] = list(cross_slice['permno'].astype(int))
    re_df['DATE'] = list(cross_slice['DATE'].astype(int))
    
    return re_df[['permno', 'DATE'] + CHARAS_LIST]



def cal_portfolio_ret(it, df):
    d, f = it[0], it[1]
    # long portfolio, qunatile 0.0~0.1; short portfolio, qunatile 0.9~1.0
    long_portfolio = df.loc[df.DATE == d][['permno', f]].sort_values(by=f, ascending=False)[:df.loc[df.DATE == d].shape[0]//10]['permno'].to_list()
    short_portfolio = df.loc[df.DATE == d][['permno', f]].sort_values(by=f, ascending=False)[-df.loc[df.DATE == d].shape[0]//10:]['permno'].to_list()
    # long-short portfolio return
    long_ret = mon_ret.loc[mon_ret.date == d].drop_duplicates('permno').set_index('permno').reindex(long_portfolio)['ret-rf'].dropna().mean()
    short_ret = mon_ret.loc[mon_ret.date == d].drop_duplicates('permno').set_index('permno').reindex(short_portfolio)['ret-rf'].dropna().mean()
    chara_ret = 0.5*(long_ret - short_ret)
    
    return chara_ret


def cal_portfolio_charas(month, df):
    mon_portfolio_chara = []
    p_name = ['p_' + chr for chr in CHARAS_LIST]
    for chr in CHARAS_LIST:
        long_portfolio = df.loc[df.DATE == month].sort_values(by=chr, ascending=False).reset_index(drop=True)[:df.loc[df.DATE == month].shape[0]//10]['permno'].to_list()
        short_portfolio = df.loc[df.DATE == month].sort_values(by=chr, ascending=False).reset_index(drop=True)[-df.loc[df.DATE == month].shape[0]//10:]['permno'].to_list()
        
        long_charas = df.loc[df.DATE == month].set_index('permno').loc[long_portfolio][CHARAS_LIST]
        short_charas = df.loc[df.DATE == month].set_index('permno').loc[short_portfolio][CHARAS_LIST]
        
        mon_portfolio_chara.append([month] + (0.5*(long_charas.mean() - short_charas.mean())).to_list())

    return pd.DataFrame(mon_portfolio_chara, index=p_name, columns=['DATE']+CHARAS_LIST)



if __name__ == '__main__':
    # pre-process share data
    processed_df = Parallel(n_jobs=-1)(delayed(pre_process)(d) for d in tqdm(datashare.DATE.drop_duplicates().to_list(), colour='green', desc='Processing'))
    processed_df = pd.concat(processed_df)

    ##TODO: calculate portfolio returns (or download preprocessed data)
    iter_list = list(product(datashare.DATE.drop_duplicates(), CHARAS_LIST))
    portfolio_rets = Parallel(n_jobs=-1)(delayed(cal_portfolio_ret)(it, df=processed_df) for it in tqdm(iter_list, colour='green', desc='Calculating'))
    portfolio_rets = pd.DataFrame(np.array(portfolio_rets).reshape(-1, 94), index=datashare.DATE.drop_duplicates(), columns=CHARAS_LIST).reset_index()
    portfolio_rets[CHARAS_LIST] = portfolio_rets[CHARAS_LIST].astype(np.float16)
    
    
    ##TODO: calculate portfolio characteristics (or download preprocessed data)
    mon_list = pd.read_pickle('data/mon_list.pkl')
    _portfolio_chara_set = Parallel(n_jobs=-1)(delayed(cal_portfolio_charas)(mon, df=processed_df) for mon in tqdm(mon_list, colour='yellow', desc='Calculating P characteristics'))
    p_charas = _portfolio_chara_set[0].copy(deep=False)
    for tdf in _portfolio_chara_set[1:]:
        p_charas = pd.concat([p_charas, tdf])
    
            
    processed_df.to_pickle('data/datashare_re.pkl')
    portfolio_rets.to_pickle('data/portfolio_rets.pkl')
    p_charas.to_pickle('data/p_charas.pkl')