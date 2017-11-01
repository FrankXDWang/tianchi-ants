import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s','--start',dest='start',default='0')
    parser.add_option('-e','--end',dest='end',default='20')
    parser.add_option('-m','--mall', dest='mall', default='')
    (options, args) = parser.parse_args(sys.argv)
    s = int(options.start)
    e = int(options.end)
    mall_id = options.mall
    file_dir = 'data/order_wifi_data/'
    model_dir = 'model/xgb_model/'
    filenames = os.listdir(file_dir)
    num = len(filenames)
    counter = 0
    filenames = [str(mall_id)+'.csv']
    
    # test
    #filenames = []
    #or mall_file in filenames:
    
    for mall_file in filenames[s:e]:
        print(mall_file+'*'*30+'\n')
        counter += 1
        print(mall_file,counter,num)
        raw_data = load_svmlight_file(file_dir+mall_file)
        train = raw_data[0].toarray()
        label = raw_data[1]
        
        #shuffle data
        permutation = np.random.permutation(label.shape[0])
        train = train[permutation,:]
        label = label[permutation]

        sz = train.shape
        kfolds = 0.9
        train_X = train[:int(sz[0] * kfolds), :]
        test_X = train[int(sz[0] * kfolds):, :]
        train_Y = label[:int(sz[0] * kfolds)]
        test_Y = label[int(sz[0] * kfolds):]

        xtrain = xgb.DMatrix(train_X, label=train_Y)
        xtest = xgb.DMatrix(test_X, label=test_Y)

        num_class = int( max(label) ) + 1

        params = {'max_depth':6, 'eta':0.1, 'num_class':num_class, 'objective':'multi:softmax','nthread':4, 'silent':0}
        num_boost_round = 30
        #watchlist = [(xtrain, 'train'),(xtest, 'test')]
        model = xgb.train(params, xtrain, num_boost_round)
        pred = model.predict(xtest)
        error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
        #print('Test acc using softmax = {}'.format(1-error_rate))
        with open('test.csv', 'a') as f:
            f.write(str(mall_file.split('.')[0])+','+str(1-error_rate)+'\n')
        model_path = model_dir+str( mall_file.split('.')[0] )+'.model'
        model.save_model(model_path)
        
