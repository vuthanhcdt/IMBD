import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

rules = {
        "NL": 5,
        "UL": 4,
        "UN": 3,
        "UR": 2,
        "NR": 1,
        "DR": 8,
        "DN": 7,
        "DL": 6,
        "NN": 0
}
#def xgboosg():
def modelfit(x_train,x_test,y_train,y_test,weight):
    from time import time
    from scipy.stats import randint as sp_randint
    from sklearn.ensemble import RandomForestClassifier
    clf = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20]}        #给定list
    #開啟超參數空間的隨機搜索

    random_search = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=1, cv=10)
    
    start=time()
    
    random_search.fit(x_train, y_train) ######@@@@@@@@@
    etr_y_predict=random_search.predict(x_test)
    

    #print("RandomizedSearchCV took %.3f seconds for %d candidates"
     #     "parameter settings."%((time()-start),n_iter_search))
    best_estimator = random_search.best_estimator_#返回最優的訓練器
    
    print("best_estimator:",best_estimator)
    print("random_search.best_score_:",random_search.best_score_)
    print(y_test)
    print(etr_y_predict)
  
   # with open('clf.pickle', 'wb') as f:
   #         pickle.dump(random_search, f)
   
    print(np.sqrt(mean_squared_error(y_test,etr_y_predict)*weight))
    return np.sqrt(mean_squared_error(y_test,etr_y_predict)*weight)
def fit(x_train,x_test,y_train,y_test):
    from time import time
    
    clf = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20]}        #给定list
    #開啟超參數空間的隨機搜索

    random_search = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=1, cv=10)
    
    
    
    random_search.fit(x_train, y_train) ######@@@@@@@@@
    etr_y_predict=random_search.predict(x_test)
    

    #print("RandomizedSearchCV took %.3f seconds for %d candidates"
     #     "parameter settings."%((time()-start),n_iter_search))
    best_estimator = random_search.best_estimator_#返回最優的訓練器
    
    print("best_estimator:",best_estimator)
    print("random_search.best_score_:",random_search.best_score_)
    print(y_test)
    print(etr_y_predict)
  
   # with open('clf.pickle', 'wb') as f:
   #         pickle.dump(random_search, f)
   
    print(np.sqrt(mean_squared_error(y_test,etr_y_predict)))
    return etr_y_predict

def run(df):
    i=1
    df = df.drop(0,axis=1)  
    df = df.drop(0,axis=0)  
    
    

    r = list(range(159, 183))
    r2 = list(range(207, 227))
    r.extend(r2)
    
    for i in r:
      k=1
      while k<len(df)+1:
        try:
            words = df[i][k].split(";")
            #print (words)
            df[i][k] = rules[words[0] + words[2]]
         #if(df[i][k][3] == ';' or df[i][k][4]==';'or df[i][k][5]==';'):
        #    df[i][k] = 87
        #    print('87')
           # count1=count1+1
        except Exception as e:
            pass
        k = k+1
  #  df.to_csv("test113.csv")
    count=1
    df = df.astype(float)
    while count<282: #把剩下的空個填入那列特徵的平均值
     #df["Input"+str(count).zfill(3)] = df["Input"+str(count).zfill(3)].fillna(df["Input"+str(count).zfill(3)].mean())
     df[count] = df[count].fillna(df[count].mean())
     count = count+1
    
    return df
def normaliz(df):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(x_scaled)
    
    return df

df_test=pd.read_csv('test.csv',header=None,sep=',')
df_train=pd.read_csv('train.csv',header=None,sep=',')
o = df_test #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。

df_test = run(df_test)
df_train =run(df_train)





dft_20= pd.concat([df_test.loc[:,144], df_test.loc[:,64],df_test.loc[:,157],df_test.loc[:,40],df_test.loc[:,65]
,df_test.loc[:,194],df_test.loc[:,121],df_test.loc[:,240],df_test.loc[:,66],df_test.loc[:,139],df_test.loc[:,20]
,df_test.loc[:,131],df_test.loc[:,63],df_test.loc[:,190],df_test.loc[:,193],df_test.loc[:,48],df_test.loc[:,202]
,df_test.loc[:,201],df_test.loc[:,61],df_test.loc[:,41]],axis=1)




df_20 = pd.concat([df_train.loc[:,144], df_train.loc[:,64],df_train.loc[:,157],df_train.loc[:,40],df_train.loc[:,65]
,df_train.loc[:,194],df_train.loc[:,121],df_train.loc[:,240],df_train.loc[:,66],df_train.loc[:,139],df_train.loc[:,20]
,df_train.loc[:,131],df_train.loc[:,63],df_train.loc[:,190],df_train.loc[:,193],df_train.loc[:,48],df_train.loc[:,202]
,df_train.loc[:,201],df_train.loc[:,61],df_train.loc[:,41]],axis=1)
    

xdf_20 = df_train.drop([144,64,157,40,65,194,121,240,66,139,20,131,63,190,193,48,202,201,61,41], axis=1)
xdft_20 = df_test.drop([144,64,157,40,65,194,121,240,66,139,20,131,63,190,193,48,202,201,61,41], axis=1)

#
def weight(column):
    all=0
    for i in column:  
        all = abs(df_train[i].mean())+all
    listh=[None for i in range(len(column)+1)]
    t=1
    for i in column:  
        listh[t] = (all-abs(df_train[i].mean()))/all
        t=t+1
    allbar=0
    for i in range(1,len(column)+1):  
        allbar = allbar+listh[i]
    bar=[None for i in range(len(column)+1)]
    for i in range(1,len(column)+1):  
        bar[i] = listh[i]/allbar
    k=0
    for i in  range(1,len(column)+1):
        k=k+bar[i]
    
    return bar




column20 = [42,86,202,20,67,33,6,240,193,89,36,134,79,109,76,16,203,9,157,127]
column20t = [144,64,157,40,65,194,121,240,66,139,20,131,63,190,193,48,202,201,61,41]
bar20 = [None for i in range(len(column20t)+1)]
bar20  =weight(column20t)




result1 = fit(xdf_20,xdft_20,df_train[144],df_test[144]) #前四十個選出來的參數
result2 = fit(xdf_20,xdft_20,df_train[64],df_test[64]) #前四十個選出來的參數
result3 = fit(xdf_20,xdft_20,df_train[157],df_test[157]) #前四十個選出來的參數
result4 = fit(xdf_20,xdft_20,df_train[40],df_test[40]) #前四十個選出來的參數
result5 = fit(xdf_20,xdft_20,df_train[65],df_test[65]) #前四十個選出來的參數
result6 = fit(xdf_20,xdft_20,df_train[194],df_test[194]) #前四十個選出來的參數
result7 = fit(xdf_20,xdft_20,df_train[121],df_test[121]) #前四十個選出來的參數
result8 = fit(xdf_20,xdft_20,df_train[240],df_test[240]) #前四十個選出來的參數
result9 = fit(xdf_20,xdft_20,df_train[66],df_test[66]) #前四十個選出來的參數
result10 = fit(xdf_20,xdft_20,df_train[139],df_test[139]) #前四十個選出來的參數
result11 = fit(xdf_20,xdft_20,df_train[20],df_test[20]) #前四十個選出來的參數
result12 = fit(xdf_20,xdft_20,df_train[131],df_test[131]) #前四十個選出來的參數
result13 = fit(xdf_20,xdft_20,df_train[63],df_test[63]) #前四十個選出來的參數
result14 = fit(xdf_20,xdft_20,df_train[190],df_test[190]) #前四十個選出來的參數
result15 = fit(xdf_20,xdft_20,df_train[193],df_test[193]) #前四十個選出來的參數
result16 = fit(xdf_20,xdft_20,df_train[48],df_test[48]) #前四十個選出來的參數
result17 = fit(xdf_20,xdft_20,df_train[202],df_test[202]) #前四十個選出來的參數
result18 = fit(xdf_20,xdft_20,df_train[201],df_test[201]) #前四十個選出來的參數
result19 = fit(xdf_20,xdft_20,df_train[61],df_test[61]) #前四十個選出來的參數
result20 = fit(xdf_20,xdft_20,df_train[41],df_test[41]) #前四十個選出來的參數
#num = np.c_[result[:,41],result[:,85],result[:,201],result[:,19],result[:,66],result[:,32],result[:,5],result[:,239],result[:,192],result[:,88],result[:,35]
#,result[:,133],result[:,78],result[:,108],result[:,75],result[:,15],result[:,202],result[:,8],result[:,156],result[:,126]]
result1 = pd.DataFrame(result1)#轉dataframe
result2 = pd.DataFrame(result2)
result3 = pd.DataFrame(result3)
result4 = pd.DataFrame(result4)
result5 = pd.DataFrame(result5)
result6 = pd.DataFrame(result6)#轉dataframe
result7 = pd.DataFrame(result7)
result8 = pd.DataFrame(result8)
result9 = pd.DataFrame(result9)
result10 = pd.DataFrame(result10)
result11 = pd.DataFrame(result11)#轉dataframe
result12 = pd.DataFrame(result12)
result13 = pd.DataFrame(result13)
result14 = pd.DataFrame(result14)
result15 = pd.DataFrame(result15)
result16 = pd.DataFrame(result16)#轉dataframe
result17 = pd.DataFrame(result17)
result18 = pd.DataFrame(result18)
result19 = pd.DataFrame(result19)
result20 = pd.DataFrame(result20)

num = pd.concat([result1, result2,result3,result4,result5,result6, result7,result8,
                 result9,result10,result11, result12,result13,result14,result15,
                 result6, result17,result18,result19,result20],axis=1) #
u=94
for i in range(0,95):
    num.rename({u: u+1}, axis='index',inplace=True)
    u=u-1
num.columns = column20t
for i  in range(1,21):
    bar20[i] = round(bar20[i],5)

all=[]
num.columns = column20t
l=1
wrmse=0
for i  in column20t:
    for j in range(1,96):
        wrmse = ((num[i][j]-dft_20[i][j])**2)*bar20[l]+wrmse
    l=l+1
#print(wrmse)
wrmse = np.sqrt(wrmse/95)
print("WRMSE: ",wrmse)

num.to_csv(r'C:\Users\fcac3\Desktop\教育部大數據\clean_data\參數預測結果.csv')
#q=pd.read_csv('new_test.csv',header=None,sep=',')
#q = q.drop([144,64,157,40,65,194,121,240,66,139,20,131,63,190,193,48,202,201,61,41], axis=1)
#q.to_csv(r'C:\Users\John\Desktop\bigdata_code\cleandata\testt.csv',header=None,index=False)