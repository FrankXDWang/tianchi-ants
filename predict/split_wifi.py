#coding=utf-8
import pandas as pd
from math import *
import pickle
import os
import sys
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s','--start',dest='start',default='0')
    parser.add_option('-e','--end',dest='end',default='100000')
    (options,args) = parser.parse_args(sys.argv)
    s = int(options.start)
    e = int(options.end)
    test = pd.read_csv('../../ants_data/AB榜测试集-evaluation_public.csv')

    # pre read wifi dict
    wifi_dir = '../wifis/'
    save_dir = 'data_/split_wifi_data/'
    wifinames = os.listdir(wifi_dir)
    _wifis = {}
    for wifiname in wifinames:
        with open(wifi_dir+wifiname,'rb') as f:
            wifis=pickle.load(f)
        _wifis[wifiname] = wifis
        
    test['shop_id'] = 's_xxx'
    test=test.drop(['user_id', 'time_stamp', 'longitude','latitude'], axis=1, inplace=False)
    # add more features into wifis here
    new_X = pd.DataFrame(columns=['row_id']+[i for i in range(10)])
    _n_samples = test.shape[0]
    for i in range(s,e):
        print(i,str(s),str(e),_n_samples)
        mall_id = test.ix[i]['mall_id']
        row_id = test.ix[i]['row_id']
        new_X.at[i,'row_id'] = row_id
        wifi_infos = test.ix[i]['wifi_infos'].split(';')
        wifis = _wifis[str(mall_id)+'_wifis.pkl']
        for j in range(len(wifi_infos)):
            _wifi = wifi_infos[j].split('|')
            if _wifi[0] in wifis:
                new_X.at[i,j]=str(wifis.index(_wifi[0]))+':'+str(_wifi[1])
    new_X.to_csv(save_dir+str(s)+'_'+str(e)+'_predict.csv',index=True,header=None)
