import xgboost as xgb
import numpy as np
import pandas as pd
dtrain = xgb.DMatrix('data/m_5085.csv')
num_class = int( max(dtrain.get_label()) ) + 1

params = {'max_depth':3,'num_class':num_class,'objective':'multi:softmax','silent':1}
num_boost_round = 3
min_merror = float("Inf")
best_params = None

for eta in [.05, .01, .005]:
    print("CV with eta={}".format(eta))
    params['eta'] = eta
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['merror'],
            early_stopping_rounds=2
          )
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMERROR {} for {} rounds\n".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = eta
print("Best params: {}, MERROR: {}".format(best_params, min_merror))

params['eta'] = best_params
model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
