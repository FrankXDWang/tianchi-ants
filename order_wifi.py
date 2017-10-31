import os
import sys
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--start", dest="start", default="0")
    parser.add_option("-e", "--end", dest="end", default="20")
    (options, args) = parser.parse_args(sys.argv)
    s = int(options.start)
    e = int(options.end)
    fileDir = 'data/split_wifi_data'
    saveDir = 'data/order_wifi_data/'
    filenames = os.listdir(fileDir)
    _counter = 0
    for filename in filenames[s:e]:
        train = pd.read_csv(fileDir+filename, header=None)
        sz = train.shape
        _counter += 1
        for i in range(sz[0]):
            print(i,sz[0],filename.split('.')[0],_counter)
            nan_number = train.ix[i,1:][train.ix[i,1:].isnull()].shape[0]
            ordered_list=sorted(train.ix[i,1:][~train.ix[i,1:].isnull()],key=lambda data: int(data.split(':')[0]))
            # filter duplicate 
            filtered_ordered_list = []
            #counter = Counter([elem.split(':')[0] for elem in ordered_list])
            counter = OrderedDict()
            for elem in ordered_list:
                key = elem.split(':')[0]
                if key not in counter.keys():
                    counter[key] = 1
                else:
                    counter[key] += 1
            for k,v in counter.items():
                _list = [elem for elem in ordered_list if elem.split(':')[0] == k]
                if v == 1:
                    filtered_ordered_list.extend(_list)
                if v > 1:
                    nan_number += v - 1
                    filtered_ordered_list.append(_list[0])
            for j in range(nan_number):
                filtered_ordered_list.append('')
            train.ix[i,1:]= filtered_ordered_list
        train.to_csv(saveDir+filename, header=None, index=False, sep =' ')
    
