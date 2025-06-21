import torch
from models.PCA import PCA
from models.FF import FF
from models.IPCA import IPCA
# from models.CA import CA0, CA1, CA2, CA3
from models.CA_wyh import CA_base_wyh,CA0_wyh,CA1_wyh,CA2_wyh,CA3_wyh,VAE_wyh,VAE_wzy
import gc
import argparse
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
from utils import *
from analysis import *
import matplotlib.pyplot as plt
from itertools import product
import os

import warnings
warnings.filterwarnings('ignore')


def predict_and_inference_non_nn(model):
    mon_list = pd.read_pickle('data/mon_list.pkl')
    test_mons = mon_list.loc[mon_list >= model.test_period[0]]
    inference_result = []
    predict_result = []
    T_bar = tqdm(test_mons.groupby(test_mons.apply(lambda x: x//10000)), colour='red', desc=f'{model.name} Inferencing & Predicting')
    
    for g in T_bar: # rolling train
        T_bar.set_postfix({'Year': g[0]})
        model.train_model()
        
        for m in g[1].to_list():
            inference_result.append(model.inference(m)) # T * N * m 
            if not len(model.omit_char):
                predict_result.append(model.predict(m))
        # model refit (change train period and valid period)
        model.refit()
    
    if not len(model.omit_char):
        inference_result = pd.DataFrame(inference_result, index=test_mons, columns=CHARAS_LIST)
        inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
        predict_result = pd.DataFrame(predict_result, index=test_mons, columns=CHARAS_LIST)
        predict_result.to_csv(f'results/predict/{model.name}_predict.csv')
    
    return inference_result

def predict_and_inference(model):
    model = model.to('cuda')
    month_list=pd.read_pickle('data/mon_list.pkl')
    test_month=month_list.loc[(month_list>model.test_period[0])]
    
    test_bar=tqdm(test_month.groupby(test_month.apply(lambda x:x//10000)),colour='red',desc=f'Testing for {model.name}')
    
    inference_result=pd.DataFrame()
    predict_result=pd.DataFrame()
    # print(test_month)
    # i=0
    for g in test_bar:
        # i+=1
        # if i>2:
        #     break
        test_bar.set_postfix({'Year':g[0]})
        
        model.reset_parameters()
        model.release_gpu()
        train_loss,val_loss=model.train_model()
        if not os.path.exists('results/loss_curve'):
            os.mkdir('results/loss_curve')
        plt.figure(figsize=(10,5))
        plt.plot(train_loss,label='train_loss')
        plt.plot(val_loss,label='val_loss')
        plt.legend()
        plt.savefig(f'results/loss_curve/{model.name}_loss_{g[0]}.png')
        plt.close()
        for m in g[1].to_list():
            m_stock_index,_,_,_=model._get_item(m)
            m_stock_index=pd.Series(m_stock_index)
            inference_R=model.inference(m)
            inference_R=inference_R.cpu().detach().numpy()
            inference_R=pd.DataFrame(inference_R,index=m_stock_index,columns=[m])
            inference_result=pd.concat([inference_result,inference_R],axis=1)
            
            predict_R=model.predict(m)
            predict_R=predict_R.cpu().detach().numpy()
            predict_R=pd.DataFrame(predict_R,index=m_stock_index,columns=[m])
            predict_result=pd.concat([predict_result,predict_R],axis=1)
            
            gc.collect()
        model.refit()
        
    inference_result=pd.DataFrame(inference_result.values.T,index=test_month,columns=CHARAS_LIST)
    predict_result=pd.DataFrame(predict_result.values.T,index=test_month,columns=CHARAS_LIST)
    inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
    predict_result.to_csv(f'results/predict/{model.name}_predict.csv')
        
    del model
    gc.collect()
    return inference_result       
    
def call_model(model_name):
    assert model_name in ['FF', 'PCA', 'IPCA', 'CA0','CA1','CA2','CA3','VAE','VAE_wzy']
    if model_name == 'FF':
        return [{
            'name': f'FF_{k}',
            'omit_char': [],
            'model': FF(K=k)
        } for k in range(1, 6)]
            
    elif model_name == 'PCA':
        return [{
            'name': f'PCA_{k}',
            'model': PCA(K=k, omit_char=[])
        } for k in range(1, 6)]
        
    elif model_name == 'IPCA':
        return [{
            'name': f'IPCA_{k}',
            'model': IPCA(K=k, omit_char=[])
        } for k in range(1, 6)]
    if model_name=='CA0':
        return[{
                'name':f'CA0_{k}',
                'model':CA0_wyh(hidden_size=k,lr=0.001,omit_char=[],device='cuda')
            }   for k in range(1,6)]
    if model_name=='CA1':
        return[{
                'name':f'CA1_{k}',
                'model':CA1_wyh(hidden_size=k,lr=0.001,omit_char=[],device='cuda')
            }   for k in range(1,6)]
    if model_name=='CA2':
        return[{
                'name':f'CA2_{k}',
                'model':CA2_wyh(hidden_size=k,lr=0.001,omit_char=[],device='cuda')
            }   for k in range(1,6)]
    if model_name=='CA3':
        return[{
                'name':f'CA3_{k}',
                'model':CA3_wyh(hidden_size=k,lr=0.001,omit_char=[],device='cuda')
            }   for k in range(1,6)]
    if model_name=='VAE':
        return[{
            'name':f'VAE_{k}',
            'model':VAE_wyh(hidden_size=k,lr=0.001,omit_char=[],device='cuda')
        }  for k in range(1,6)
        ]
    if model_name=='VAE_wzy':
        return[{
            'name':f'VAE_wzy_{k}',
            'model':VAE_wzy(hidden_size=k,lr=0.001,omit_char=[],device='cuda')
        }  for k in range(4,6)
        ]        
         
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='VAE_wzy')
    parser.add_argument('--omit_char', type=str, default='')
    args=parser.parse_args()
    
    if 'results' not in os.listdir('./'):
        os.mkdir('results')
    if 'inference' not in os.listdir('./results'):
        os.mkdir('results/inference')
    if 'predict' not in os.listdir('./results'):
        os.mkdir('results/predict')
    if 'imgs' not in os.listdir('./'):
        os.mkdir('imgs')
    R_square=[]
    models_name=[]
    model_list=args.model.split(',')
    for model_name in model_list:
        model_ks=call_model(model_name)
        for model in model_ks:
            models_name.append(model['name'])
            # print(model['model'])
            if ('VAE' in model['name']) or (model['name'].split('_')[0][:-1] == 'CA'):
                model['model'].to("cuda")
                inference_result=predict_and_inference(model['model'])
                print('correct')
            else:
                inference_result=predict_and_inference_non_nn(model['model'])
                print('wrong')
            R_square.append(calculate_R2(model['model'], 'inference'))
            alpha_plot(model['model'], 'inference', save_dir='imgs')
                
            
    p = time.localtime()
    time_str = "{:0>4d}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}".format(p.tm_year, p.tm_mon, p.tm_mday, p.tm_hour, p.tm_min, p.tm_sec)
    filename = f"R_squares/{time_str}.json"
    obj = {
        "models": models_name,
        'omit_char': args.omit_char.split(' '),
        "R2_total": R_square,
    }

    with open(filename, "w") as out_file:
        json.dump(obj, out_file)