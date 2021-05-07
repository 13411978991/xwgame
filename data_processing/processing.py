import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.signal import resample
from sklearn import metrics
import random
import pickle

#聚合函数
def get_dic(df, main_col, fea_col, agg):
    dic = df.groupby(main_col)[fea_col].agg(agg).to_dict()
    fea_name = '_'.join([main_col, fea_col, agg])
    return fea_name, dic

#返回统计特征
def get_1st_order_xyz_features(df, fea_cols, main_col='fragment_id'):
    df_fea = pd.DataFrame()
    df_fea[main_col] = df[main_col].unique()
    ## count 特征 ##
    _, dic = get_dic(df, main_col, fea_cols[0], 'count')  # 统计采样点个数
    df_fea['cnt'] = df_fea[main_col].map(dic).values

    ## 数值统计特征 ##
    for f in tqdm(fea_cols):
        for agg in ['min', 'max', 'mean', 'std', 'skew', 'median']:
            fea_name, dic = get_dic(df, main_col, f, agg)
            df_fea[fea_name] = df_fea[main_col].map(dic).values
        df_fea['_'.join([main_col, f, 'gap'])] = df_fea['_'.join([main_col, f, 'max'])] - df_fea[
            '_'.join([main_col, f, 'min'])]
        dic = df.groupby(main_col)[f].quantile(.1).to_dict()
        df_fea['_'.join([main_col, f, '_qu10'])] = df_fea[main_col].map(dic).values
        dic = df.groupby(main_col)[f].quantile(.90).to_dict()
        df_fea['_'.join([main_col, f, '_qu90'])] = df_fea[main_col].map(dic).values

    return df_fea


#返回序列特征
def feature(train,simple=True,acc=False):#输入数据，简单特征or复杂特征，是否衍生加速度
    train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5#x,y,z的向量模
    train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5#xg,yg,zg的向量模

    #衍生加速度
    if acc:
        clo=[]
        tf_diff=train
        try:
          tf_diff=tf_diff.drop(['behavior_id'],axis=1).groupby('fragment_id').diff().fillna(0)
        except:
          tf_diff=tf_diff.groupby('fragment_id').diff().fillna(0)
        for i in tf_diff.drop('time_point',axis=1).columns:
          tf_diff[i]=tf_diff[i]/tf_diff.time_point
        tf_diff=tf_diff.fillna(0).drop('time_point',axis=1)
        try:
          for i in train.drop(['fragment_id','time_point','behavior_id'],1).columns:
            clo.append(i+'_diff')
          tf_diff.columns=clo
        except:
          for i in train.drop(['fragment_id','time_point'],1).columns:
            clo.append(i+'_diff')
          tf_diff.columns=clo

        train=pd.concat([train,tf_diff],1)

    #复杂特征
    if not simple:
        train['mod_xy'] = (train.acc_x ** 2 + train.acc_y ** 2 ) ** .5
        train['mod_xz'] = (train.acc_x ** 2 + train.acc_z ** 2 ) ** .5
        train['mod_zy'] = (train.acc_z ** 2 + train.acc_y ** 2 ) ** .5
        train['modg_xy'] = (train.acc_xg ** 2 + train.acc_yg ** 2 ) ** .5
        train['modg_xz'] = (train.acc_xg ** 2 + train.acc_zg ** 2 ) ** .5
        train['modg_zy'] = (train.acc_zg ** 2 + train.acc_yg ** 2 ) ** .5
        train['diff_xy']=train.acc_x-train.acc_y
        train['diff_xz']=train.acc_x-train.acc_z
        train['diff_yz']=train.acc_y-train.acc_z
        train['diff_xyg']=train.acc_xg-train.acc_yg
        train['diff_xzg']=train.acc_xg-train.acc_zg
        train['diff_yzg']=train.acc_yg-train.acc_zg
    try:
        origin_fea_cols = train.drop(['fragment_id','time_point','behavior_id'],axis=1).columns.tolist()
    except:
        origin_fea_cols = train.drop(['fragment_id','time_point'],axis=1).columns.tolist()


    df_xyz_fea1 = get_1st_order_xyz_features(train, origin_fea_cols, main_col='fragment_id')#输出统计特征
    del df_xyz_fea1['cnt']

    #将统计特征和序列特征打包进x
    sha = train.fragment_id.drop_duplicates().shape[0]
    if 'behavior_id' in train.columns.tolist():
        x = (np.zeros((sha, train.shape[1] - 3, 61)), [])
    else:
        x = (np.zeros((sha, train.shape[1] - 2, 61)), [])
    flag = 0
    for i in tqdm(train.fragment_id.drop_duplicates().tolist()):
        tmp = train[train.fragment_id == i][:61]
        tmp1 = df_xyz_fea1[df_xyz_fea1.fragment_id == i].values[0][1:]

        try:
            x[0][flag, 0:tmp.shape[1], 0:tmp.shape[0]] = tmp.drop(['fragment_id', 'time_point', 'behavior_id'],axis=1).values.T
        except:

            x[0][flag, 0:tmp.shape[1], 0:tmp.shape[0]] = tmp.drop(['fragment_id', 'time_point'], axis=1).values.T
        x[1].append(tmp1)
        flag += 1
    try:
        y = train.groupby('fragment_id')['behavior_id'].min().tolist()
    except:
        return x

    return x, y#返回data,label,格式x:([dim1,dim2,dim3],[dim1,dim4]),y:[dim1]



#数据增强
clo = ['acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg']#仅增强原始数据序列
def st(train, f):
    df = pd.DataFrame()
    for c in tqdm(clo):
        dic = train.groupby('fragment_id')[c].std().tolist()
        dic_cnt = train.groupby('fragment_id')[c].count().tolist()
        flag = 0
        for i in range(len(dic)):
            x = np.random.normal(0, dic[i] / 4, dic_cnt[i])#按数据方差为mu*1/4添加高斯噪声
            if flag == 0:
                flag = 1
                lst = x
            else:
                lst = np.hstack((lst, x))

        df[c] = train[c] + lst

        df[['fragment_id', 'time_point', 'behavior_id']] = train[['fragment_id', 'time_point', 'behavior_id']]
    df['fragment_id'] += 7292 * (f + 1)
    return df


def strong(train, f):  # f为增强的倍数
    tf = train
    for i in range(f):
        df = st(tf, i)
        train = pd.concat([train, df], 0)
    return train
# train_data_=strong(train_data_,5)

