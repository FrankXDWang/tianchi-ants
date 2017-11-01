import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s','--start',dest='start',default='0')
    parser.add_option('-e','--end',dest='end',default='20')
    
    # test for single model tuning
    parser.add_option('-m','--mall', dest='mall', default='')
    (options, args) = parser.parse_args(sys.argv)
    s = int(options.start)
    e = int(options.end)
    
    mall_id = options.mall
    
    file_dir = 'data/order_wifi_data/'
    tll_dir = 'data/tll_data/'
    model_dir = 'model/lgb_model/'
    filenames = os.listdir(file_dir)
    num = len(filenames)
    counter = 0
    
    # test
    filenames = [str(mall_id)+'.csv']
    for mall_file in filenames:
    #for mall_file in filenames[s:e]:
        counter += 1
        print(counter,num)
        raw_data = load_svmlight_file(file_dir+mall_file)
        train = raw_data[0].toarray()
        label = raw_data[1]
        
        # additional features including longitude, latitude and time stamp
        tll_data = pd.read_csv(tll_dir+mall_file, header=None)
        train = np.hstack((train, tll_data.ix[:,1:]))
        
        # shuffle data
        np.random.seed(3)
        permutation = np.random.permutation(label.shape[0])
        train = train[permutation,:]
        label = label[permutation]

        sz = train.shape
        kfolds = 0.8
        train_X = train[:int(sz[0] * kfolds), :]
        train_Y = label[:int(sz[0] * kfolds)]
        
        val_X = train[int(sz[0] * kfolds) : int(sz[0] * (kfolds + 0.1)), :]
        val_Y = label[int(sz[0] * kfolds) : int( sz[0] * (kfolds + 0.1) )]
        
        test_X = train[int(sz[0] * (kfolds + 0.1 )):, :]
        test_Y = label[int( sz[0] * (kfolds + 0.1) ):]

        ltrain = lgb.Dataset(train_X, label=train_Y)
        lval = lgb.Dataset(val_X, label=val_Y)

        num_class = int( max(label) ) + 1

        params = {  'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'metric': {'multi_error'},
                    'num_class': num_class,
                    'num_leaves': 31,
                    'learning_rate': 0.02,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 4,
                    'verbose':0,
                    #'device': 'gpu'
            }
        num_boost_round = 70
        model = lgb.train(  params,
                            ltrain,
                            num_boost_round,
                            valid_sets=lval,
                            early_stopping_rounds=5
                            )
        y_pred = model.predict(test_X, num_iteration=model.best_iteration)

        acc = accuracy_score(test_Y, np.argmax(y_pred, axis=1))
        print('Val acc is {}'.format(acc))
        #with open('test.csv', 'a') as f:
        #    f.write(str(mall_file.split('.')[0])+','+str(acc)+'\n')
        model_path = model_dir+str( mall_file.split('.')[0] )+'.model'
        model.save_model(model_path, num_iteration=model.best_iteration)        
