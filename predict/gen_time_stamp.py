import pandas as pd
data = pd.read_csv('../../ants_data/AB榜测试集-evaluation_public.csv')
time_stamps = [data.ix[:,'time_stamp'][i].split(' ')[1].split(':')[0] for i in range(data.shape[0])]
_time_stamps = pd.DataFrame(time_stamps)
rst = pd.concat((data.ix[:, 'row_id'],_time_stamps), axis=1)
rst.columns = ['row_id','time_stamp']
rst.to_csv('time_stamp.csv',index=False)
