import pickle
import pandas as pd
from datetime import datetime

#train = pd.read_csv('../ants_data/训练数据-ccf_first_round_user_shop_behavior.csv')
#test = pd.read_csv('../ants_data/AB榜测试集-evaluation_public.csv')

# get dict for shop and wifi
#shop_wifi_split = [ (shop_id, [wifi.split('|') for wifi in wifi_info.split(';')]) for shop_id,wifi_info in train[['shop_id','wifi_infos']].values]

#shop_wifi_dict = {}
#for shop_id, bssids in shop_wifi_split:
#    for bssid in bssids:
#        if bssid[2] == 'true':
#            shop_wifi_dict[bssid[0]] = shop_id
#            break
#print(len(shop_wifi_dict))

with open('shop_wifi_dict.pkl', 'rb') as f:
    shop_wifi_dict = pickle.load(f)

#test_results = [ (row_id, [wifi.split('|') for wifi in wifi_info.split(';')]) for row_id,wifi_info in test[['row_id','wifi_infos']].values]

#predict_rowid_shop = {}
#for row_id, bssids in test_results:
#    for bssid in bssids:
#        if bssid[2] == 'true' and bssid[0] in shop_wifi_dict.keys():
#            predict_rowid_shop[row_id] = shop_wifi_dict[bssid[0]]
#            break
#print(len(predict_rowid_shop))
with open('predict_rowid_shop.pkl', 'rb') as f:
    predict_rowid_shop = pickle.load(f)

def correct(row_id, shop_id, predict_rowid_shop):
    if row_id in predict_rowid_shop.keys():
        return predict_rowid_shop[row_id]   
    else:
        return shop_id

submit = pd.read_csv('../submits/sub.csv')
test_num = submit.shape[0]
not_same = []
#test_num = 10
tic = datetime.now()
for i in range(test_num):
    print(i,test_num)
    row_id = submit.ix[i]['row_id']
    shop_id = submit.ix[i]['shop_id']
    correct_shop_id = correct(row_id, shop_id,predict_rowid_shop)
    submit.loc[i,'shop_id'] = correct_shop_id
    
    # really fast opr!
    #submit.set_value(i,'shop_id',correct_shop_id) 
    
    if shop_id != correct_shop_id:
        print('OOps!',shop_id, submit.loc[i]['shop_id'])
        not_same.append(row_id)
        break
toc = datetime.now()
print('Elapsed time:{}'.format((toc - tic).total_seconds()))
#submit.to_csv('result.csv', index=False)

