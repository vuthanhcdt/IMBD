# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

header = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
# %%
#讀取與處理訓練集數據
path = './data/train.csv'

drop_columns = ['SeqNo','O']

data = pd.read_csv(path, engine ='python',index_col=None)
label = data['O']
data = data.drop(drop_columns,axis=1)

# %%
#模型訓練
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(data,label)

# %%
#讀取與處理測試集數據
test_path = './data/test.csv'
test_data = pd.read_csv(test_path, engine ='python',index_col=None)
test_label = test_data['O']
test_data = test_data.drop(drop_columns,axis=1)

# %%
#預測
pred_y = lgb_model.predict(test_data)

#mae
mae = mean_absolute_error(test_label, pred_y)
print('mae = ',mae)#3.528

resultPath = './pred_result/'
np.savetxt(resultPath+'testdata_Step'+str(1)+'.csv', pred_y, delimiter=',')
