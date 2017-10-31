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
    pass