#coding=utf-8
import pickle
import pandas as pd
from math import *
from multiprocessing import *

def calcDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    """
        # input Lat_A 纬度A
        # input Lng_A 经度A
        # input Lat_B 纬度B
        # input Lng_B 经度B
        # output distance 距离(km)
    """
    ra = 6378.140  # 赤道半径 (km)
    rb = 6356.755  # 极半径 (km)
    flatten = (ra - rb) / ra  # 地球扁率
    rad_lat_A, rad_lng_A, rad_lat_B, rad_lng_B = map(radians, [Lat_A, Lng_A, Lat_B, Lng_B])  
    pA = atan(rb / ra * tan(rad_lat_A))
    pB = atan(rb / ra * tan(rad_lat_B))
    xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
    c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
    c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (xx + dr)
    return distance

def haversine(lon1, lat1, lon2, lat2):  
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 
    return c * r * 1000

def get_nearest_shop_id(longitude, latitude, shop_ids_by_mall_id):
    #map_data = map(lambda lat, lng: calcDistance(lat, lng, latitude, longitude), shop_ids_by_mall_id['latitude'], shop_ids_by_mall_id['longitude'])
    map_data = map(lambda lng, lat: haversine(lng, lat, longitude, latitude), shop_ids_by_mall_id['longitude'], shop_ids_by_mall_id['latitude'])
    data = list(map_data)
    shop_ids_by_mall_id = shop_ids_by_mall_id.reset_index(drop=True)
    return min(data),shop_ids_by_mall_id.ix[data.index(min(data))]['shop_id']

def do(i, shop, test): 
        mall_id = test.ix[i]['mall_id']
        shop_ids_by_mall_id = shop[shop.mall_id == mall_id]
        min_dist, nn_shop_id = get_nearest_shop_id(test.ix[i]['longitude'], test.ix[i]['latitude'], shop_ids_by_mall_id)
        test.loc[i,'shop_id']= nn_shop_id
        print ("sample_num: ", i, "/", n_samples, " min_dist:", min_dist, " shop_id:", test.ix[i]['shop_id'])

if __name__ == '__main__':
    ants_dir = "../ants_data/"
    shop = pd.read_csv(ants_dir+'训练数据-ccf_first_round_shop_info.csv')
    train = pd.read_csv(ants_dir+'训练数据-ccf_first_round_user_shop_behavior.csv')
    test = pd.read_csv(ants_dir+'AB榜测试集-evaluation_public.csv')
    test['shop_id']='s_xxx'
    mall_ids = list( shop.ix[:,'mall_id'].unique() )
    for mall_id in mall_ids:
        shop_ids_by_mall_id = shop[shop.mall_id == mall_id]
        
        #print(shop_ids_by_mall_id)
        data_with_mall_id = pd.merge(shop_ids_by_mall_id, train.ix[:,['wifi_infos','shop_id']],on='shop_id')
        
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
        
        data_with_mall_id.drop(['category_id','longitude','latitude','price','mall_id'], axis=1, inplace=True)
        
        # add more features into wifis here
        new_X = pd.DataFrame(columns=range(10))
        new_Y = list()
        _n_samples = data_with_mall_id.shape[0]
        #_n_samples = 10
        for i in range(_n_samples):
            print(i,_n_samples)
            shop_id = data_with_mall_id.ix[i]['shop_id']
            new_Y.append(shops.index(shop_id))
            wifi_infos = data_with_mall_id.ix[i]['wifi_infos'].split(';')
            for j in range(len(wifi_infos)):
                _wifi = wifi_infos[j].split('|')
                new_X.at[i,j]=str(wifis.index(_wifi[0]))+':'+str(_wifi[1])
        new_Y = pd.DataFrame(new_Y,columns=['shop_id'])
        train_data_by_mall_id = pd.concat([new_Y, new_X], axis=1)
        # shuffle samples
        train_data_by_mall_id=train_data_by_mall_id.sample(frac=1)
        train_data_by_mall_id.to_csv("data/"+str(mall_id)+'.csv',index=False,header=None)
        break
