import os
import pandas as pd
result_dir = 'results/'
#filenames = os.listdir(result_dir)
filenames = ['0_60000.csv','60000_120000.csv','120000_180000.csv', '180000_240000.csv', '240000_300000.csv', '300000_360000.csv', '360000_420000.csv', '420000_483931.csv']
result = pd.DataFrame()

for filename in filenames:
    data = pd.read_csv(result_dir+filename, header=None)
    result = pd.concat([result, data], axis=0)
result.columns = ['row_id', 'shop_id']
result.to_csv('submits/result.csv', index=False)
