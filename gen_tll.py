import pickle
import pandas as pd

if __name__ == '__main__':
    ants_dir = "../ants_data/"
    shop = pd.read_csv(ants_dir+'训练数据-ccf_first_round_shop_info.csv')
    train = pd.read_csv(ants_dir+'训练数据-ccf_first_round_user_shop_behavior.csv')
    mall_ids = list( shop.ix[:,'mall_id'].unique() )
    
    for mall_id in mall_ids:
        shop_ids_by_mall_id = shop[shop.mall_id == mall_id]
        
        #print(shop_ids_by_mall_id)
        data_with_mall_id = pd.merge(shop_ids_by_mall_id, train.ix[:,['wifi_infos','shop_id','time_stamp','longitude','latitude']],on='shop_id')
        
        # get all shops
        shops = list(data_with_mall_id.ix[:,'shop_id'].unique())
        
        # get all wifis
        wifis = set()
        for wifi in data_with_mall_id.ix[:,'wifi_infos']:
            for _wifi in wifi.split(';'):
                wifis.add(_wifi.split('|')[0])
        wifis = list(wifis)
    
        with open('shops/'+str(mall_id)+'_shops.pkl','wb') as f:
                pickle.dump(shops,f)
        with open('wifis/'+str(mall_id)+'_wifis.pkl','wb') as f:
                pickle.dump(wifis,f)
                
        #with open('shops/'+str(mall_id)+'_shops.pkl','rb') as f:
        #        shops = pickle.load(f)
        #        print(shops)
        
        data_with_mall_id.drop(['category_id','longitude_x','latitude_x','price','mall_id'], axis=1, inplace=True)
        
        num_data_with_mall_id = data_with_mall_id.shape[0]
        time_stamps = [data_with_mall_id.ix[:,'time_stamp'][i].split(' ')[1].split(':')[0] for i in range(num_data_with_mall_id)]
        _time_stamps = pd.DataFrame(time_stamps)
        rst = pd.concat((data_with_mall_id.ix[:, ['shop_id','longitude_y', 'latitude_y']],_time_stamps), axis=1)
        rst.columns = ['shop_id', 'longitude_y', 'latitude_y', 'time_stamp']
        rst.to_csv("tll_data/"+str(mall_id)+'.csv',index=False,header=None)
