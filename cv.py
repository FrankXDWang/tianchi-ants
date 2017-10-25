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

params = {'max_depth':5, 'num_class':num_class, 'objective':'multi:softmax','nthread':4, 'silent':0}
num_boost_round = 10
watchlist = [(xtrain, 'train'),(xtest, 'test')]
min_merror = float("Inf")
best_params = None
for eta in [.1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    params['eta'] = eta
    cv_results = xgb.cv(
            params,
            xtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['merror'],
            early_stopping_rounds=3
          )
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMERROR {} for {} rounds\n".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = eta

params['eta'] = best_params
model = xgb.train(params, xtrain, num_boost_round, watchlist, early_stopping_rounds=3)
model_path = str( mall_file.split('.')[0] )+'.model'
model.save_model(model_path)
model = xgb.Booster()
model.load_model(model_path)
pred = model.predict(xtest)
print(pred)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))
