import xgboost as xgb
import numpy as np
from sklearn.datasets import load_svmlight_file

mall_file = 'm_5085.csv'
raw_data = load_svmlight_file(mall_file)
train = raw_data[0].toarray()
label = raw_data[1]
sz = train.shape
kfolds = 0.7
train_X = train[:int(sz[0] * kfolds), :]
test_X = train[int(sz[0] * kfolds):, :]
train_Y = label[:int(sz[0] * kfolds)]
test_Y = label[int(sz[0] * kfolds):]

xtrain = xgb.DMatrix(train_X, label=train_Y)
xtest = xgb.DMatrix(test_X, label=test_Y)

num_class = int( max(xtrain.get_label()) ) + 1

params = {'max_depth':3, 'eta':0.1, 'num_class':num_class, 'objective':'multi:softmax','nthread':4, 'silent':0}
num_boost_round = 3
watchlist = [(xtrain, 'train'),(xtest, 'test')]
#model = xgb.train(params, xtrain, num_boost_round, watchlist)
#pred = model.predict(xtest)
#error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
#print('Test error using softmax = {}'.format(error_rate))

model_path = str( mall_file.split('.')[0] )+'.model'
#model.save_model(model_path)
model = xgb.Booster()
model.load_model(model_path)

pred = model.predict(xtest)
print(pred)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

#min_merror = float("Inf")
#best_params = None
#for eta in [.05, .01, .005]:
#    print("CV with eta={}".format(eta))
#    params['eta'] = eta
#    cv_results = xgb.cv(
#            params,
#            xtrain,
#            num_boost_round=num_boost_round,
#            seed=42,
#            nfold=3,
#            metrics=['merror'],
#            early_stopping_rounds=2
#          )
#    mean_merror = cv_results['test-merror-mean'].min()
#    boost_rounds = cv_results['test-merror-mean'].argmin()
#    print("\tMERROR {} for {} rounds\n".format(mean_merror, boost_rounds))
#    if mean_merror < min_merror:
#        min_merror = mean_merror
#        best_params = eta
#print("Best params: {}, MERROR: {}".format(best_params, min_merror))

