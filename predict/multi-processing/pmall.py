import os
import sys
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import time
from optparse import OptionParser
from sklearn.datasets import load_svmlight_file
from multiprocessing import Pool

def do(s, e):
    """
        group mall by ordered test data from s to e
        s: start
        e: end
    """
    # read test data to get mall_id
    test = pd.read_csv('../../../ants_data/AB榜测试集-evaluation_public.csv')
    mall_ids = test.ix[:,['row_id','mall_id']]
     # pre read wifi dict
    wifi_dir = '../../wifis/'
    wifinames = os.listdir(wifi_dir)
    _wifis = {}
    for wifiname in wifinames:
        with open(wifi_dir+wifiname,'rb') as f:
            wifis=pickle.load(f)
        _wifis['m_'+str(wifiname.split('_')[1])] = wifis
        
    sample_file = "../data_/order_wifi_data/"+str(s)+'_'+str(e)+'_ordered_predict.csv'
    raw_data = pd.read_csv(sample_file, header=None, sep = ' ', skip_blank_lines=False)
    block = mall_ids[s:e] 
    block = block.reset_index(drop=True)
    num = block.shape[0]
    for i in range(num):
        print(i) 
        mall_id = block.ix[i]['mall_id']
        row_id = block.ix[i]['row_id']
        wifi_max = len(_wifis[mall_id])-1
        # default ascending for wifi info
        not_nan_wifi=raw_data.ix[i][1:][~raw_data.ix[i][1:].isnull()]
        wifis_info = str(' '.join(not_nan_wifi))
        if '0' not in [idx.split(':')[0] for idx in not_nan_wifi]:
            wifis_info = '0:-999'+ ' '+wifis_info

        if str(wifi_max) not in [idx.split(':')[0] for idx in not_nan_wifi]:
            wifis_info = wifis_info+' '+str(wifi_max)+':-999'
        
        with open('malls/'+str(mall_id)+'.csv', 'a') as f:
            f.write(str(row_id)+' '+wifis_info+'\n')

if __name__ == '__main__':
    p = Pool()
    total_start = 0
    total_end = 483931
    step = 60000
    start = time.time()
    for x in range(0, int(total_end/60000)):
        s = x * 60000
        if x == 7:
            e = s + 63931
        else:
            e = s + 60000
        p.apply_async(do, args=(s, e,))
    p.close()
    p.join()
    end = time.time()
    print('time:{}'.format(end-start))
    print('Over!')

