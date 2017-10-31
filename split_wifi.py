#coding=utf-8
import sys
import pickle
import pandas as pd
from math import *
from multiprocessing import *
from optparse import OptionParser

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("-s", "--start", dest="start", default="0")
    parser.add_option("-e", "--end", dest="end", default="20")
    (options, args) = parser.parse_args(sys.argv)
    s = int(options.start)
    e = int(options.end)
    ants_dir = "../ants_data/"
    shop = pd.read_csv(ants_dir+'训练数据-ccf_first_round_shop_info.csv')
    train = pd.read_csv(ants_dir+'训练数据-ccf_first_round_user_shop_behavior.csv')
    test = pd.read_csv(ants_dir+'AB榜测试集-evaluation_public.csv')
    test['shop_id']='s_xxx'
    mall_ids = list( shop.ix[:,'mall_id'].unique() )
    for mall_id in mall_ids[s:e]:
        shop_ids_by_mall_id = shop[shop.mall_id == mall_id]
        
        #print(shop_ids_by_mall_id)
        data_with_mall_id = pd.merge(shop_ids_by_mall_id, train.ix[:,['wifi_infos','shop_id']],on='shop_id')
        
        # get all shops
        shops = list(data_with_mall_id.ix[:,'shop_id'].unique())
        
        # get all wifis
        wifis = set()
        for wifi in data_with_mall_id.ix[:,'wifi_infos']:
            for _wifi in wifi.split(';'):
                wifis.add(_wifi.split('|')[0])
        wifis = list(wifis)
    
        with open('shops/'+str(mall_id)+'_shops.pkl','wb') as f:
                pickle.dump(shops,f)
        with open('wifis/'+str(mall_id)+'_wifis.pkl','wb') as f:
                pickle.dump(wifis,f)
                
        #with open('shops/'+str(mall_id)+'_shops.pkl','rb') as f:
        #        shops = pickle.load(f)
        #        print(shops)
        
        data_with_mall_id.drop(['category_id','longitude','latitude','price','mall_id'], axis=1, inplace=True)
        
        # add more features into wifis here
        new_X = pd.DataFrame(columns=range(10))
        new_Y = list()
        _n_samples = data_with_mall_id.shape[0]
        #_n_samples = 10
        for i in range(_n_samples):
            print(i,_n_samples)
            shop_id = data_with_mall_id.ix[i]['shop_id']
            new_Y.append(shops.index(shop_id))
            wifi_infos = data_with_mall_id.ix[i]['wifi_infos'].split(';')
            for j in range(len(wifi_infos)):
                _wifi = wifi_infos[j].split('|')
                new_X.at[i,j]=str(wifis.index(_wifi[0]))+':'+str(_wifi[1])
        new_Y = pd.DataFrame(new_Y,columns=['shop_id'])
        train_data_by_mall_id = pd.concat([new_Y, new_X], axis=1)
        
        # shuffle samples
        #train_data_by_mall_id=train_data_by_mall_id.sample(frac=1)
        
        train_data_by_mall_id.to_csv("data/split_wifi_data/"+str(mall_id)+'.csv',index=False,header=None)
        
