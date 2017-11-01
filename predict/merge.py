import os
import pandas as pd
result_dir = 'results/'
filenames = os.listdir(result_dir)
result = pd.DataFrame()

for filename in filenames:
    data = pd.read_csv(result_dir+filename)
    result = pd.concat([result, data], axis=0)
#result.columns = ['row_id', 'shop_id']
template = pd.read_csv('result_template.csv')

# reorder the row_id as template
row_ids = template.ix[:,'row_id'].tolist()
result['row_id'] = result['row_id'].astype('category')
result['row_id'].cat.reorder_categories(row_ids, inplace=True)
result.sort_values('row_id', inplace=True)
result.to_csv('submits/result.csv', index=False)
