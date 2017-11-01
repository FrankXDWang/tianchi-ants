import os
import sys
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from optparse import OptionParser
from sklearn.datasets import load_svmlight_file

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option('-s','--start',dest='start',default='0')
    parser.add_option('-e','--end',dest='end',default='20')
    (options, args) = parser.parse_args(sys.argv)
    s = int(options.start)
    e = int(options.end)
    file_dir = 'malls/'
    model_dir = '../model/lgb_model/'
    res_dir = 'results/'
    shop_dir = '../shops/'

    # preload all models and shops
    model_dict = {}
    model_shop_dict = {}
    models = os.listdir(model_dir)
    for model_path in models:
        model = lgb.Booster(model_file=model_dir+model_path)
        mall_id = model_path.split('.')[0]
        model_dict[mall_id] = model
        with open(shop_dir+mall_id+'_shops.pkl', 'rb') as f:
            model_shop_dict[mall_id] = pickle.load(f)
    
    # preload all longitude,latitude and time_stamp
    add_feature = pd.read_csv('time_stamp.csv') 
    malls = os.listdir(file_dir)
    for mall in malls[s:e]:
        raw_data = load_svmlight_file(file_dir+mall)
        testX = raw_data[0].toarray()
        testY = raw_data[1]
        testY = testY.astype('int')
        _add_feature = add_feature[add_feature['row_id'].isin(testY.tolist())]
        _add_feature = np.array(_add_feature)
        _testX = np.hstack((testX, _add_feature[:,1:]))
        mall_id = mall.split('.')[0]
        pred = model_dict[mall_id].predict(_testX)
        _pred = np.argmax(pred, axis=1)
        shop_id = [model_shop_dict[mall_id][pred] for pred in _pred]
        
        _shop_id = np.array(shop_id)
        _shop_id = _shop_id.reshape((_shop_id.shape[0],1))
        testY = testY.reshape((testY.shape[0],1))
        rst = np.hstack((testY, _shop_id))
        _rst = pd.DataFrame(rst, columns = ['row_id','shop_id'])
        _rst.to_csv(res_dir+mall, index=False)
