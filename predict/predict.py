import os
import sys
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from optparse import OptionParser
from sklearn.datasets import load_svmlight_file

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option('-s','--start',dest='start',default='0')
    parser.add_option('-e','--end',dest='end',default='60000')
    (options, args) = parser.parse_args(sys.argv)
    s = int(options.start)
    e = int(options.end)
    file_dir = 'ordered_predict_data/'
    model_dir = 'model/'
    res_dir = 'results/'
    shop_dir = '../shops/'

    # preload all models and shops
    model_dict = {}
    model_shop_dict = {}
    models = os.listdir(model_dir)
    for model_path in models:
        model = xgb.Booster()
        model.load_model(model_dir+model_path)
        mall_id = model_path.split('.')[0]
        model_dict[mall_id] = model
        with open(shop_dir+mall_id+'_shops.pkl', 'rb') as f:
            model_shop_dict[mall_id] = pickle.load(f)

    # read test data to get mall_id
    test = pd.read_csv('AB榜测试集-evaluation_public.csv')
    mall_ids = test.ix[:,['row_id','mall_id']]
    test_to_predict_file = str(s)+'_'+str(e)+'_ordered_predict.csv'
    results_file = res_dir+str(s)+'_'+str(e)+'.csv'
    sample_file = file_dir + test_to_predict_file
    #sample_file = '_test.csv'
    raw_data = load_svmlight_file(sample_file)
    testX = raw_data[0].toarray()
    testY = raw_data[1]

    block = mall_ids[s:e]
    block = block.reset_index(drop=True)
    num = block.shape[0]
    #num = 2
    for i in range(num):
        print(i) 
        _testX = testX[i].reshape( ( 1, testX[i].shape[0] ) )
        sample = xgb.DMatrix(_testX, label=np.array( [testY[i]] ) )
        row_id = block.ix[i]['row_id']
        mall_id = block.ix[i]['mall_id']
        
        pred = model_dict[mall_id].predict(sample)
        shop_id = model_shop_dict[mall_id][int(pred[0])]
        with open(results_file, 'a') as f:
            f.write(str(row_id)+','+str(shop_id)+'\n')

