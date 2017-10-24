import pandas as pd
import numpy as np
import os
fileDir = 'data/'
saveDir = 'ordered_data/'
filenames = os.listdir(fileDir)
counter = 0
for filename in filenames:
    train = pd.read_csv(fileDir+filename, header=None)
    sz = train.shape
    for i in range(sz[0]):
        print(i,sz[0],counter,filename)
        nan_number = train.ix[i,1:][train.ix[i,1:].isnull()].shape[0]
        ordered_list=sorted(train.ix[i,1:][~train.ix[i,1:].isnull()],key=lambda data: int(data.split(':')[0]))
        for j in range(nan_number):
            ordered_list.append('')
        train.at[i,1:]= ordered_list
    train.to_csv(saveDir+filename, header=None, index=False, sep =' ')
    
