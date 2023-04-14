import random
import torch.utils.data as torch_data
import torch as th
import scipy.io as scio
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def Loader(data_path):
    data = scio.loadmat(data_path)
    scaler = StandardScaler()
    rawdatas = data.get('rawdata_org')
    df = pd.read_csv(data_path.replace('Datas.mat','2021-01.csv')).T.index.values.tolist()
    df.remove('sasdate')
    CPI_index = df.index('CPIAUCSL')

    scaler.fit(rawdatas)
    rawdatas = scaler.transform(rawdatas)

    driven_datas = np.concatenate((rawdatas[:,:CPI_index], rawdatas[:,CPI_index+1:]),-1)
    target_datas = rawdatas[:,CPI_index:CPI_index+1]

    df = pd.read_csv(data_path.replace('Datas.mat','Group.csv'))
    df_list = df.T.index.values.tolist()
    df_list.remove('sasdate')
    CPI_index = df_list.index('CPIAUCSL')
    groups = np.concatenate((df.T.values[1:CPI_index + 1, 0], df.T.values[CPI_index + 1 + 1:, 0]), -1).tolist()
    group0,group1,group2,group3,group4,group5,group6,group7 = [],[],[],[],[],[],[],[]

    df_list_group0,df_list_group1,df_list_group2,df_list_group3,df_list_group4,df_list_group5,df_list_group6,df_list_group7 = [],[],[],[],[],[],[],[]
    df_list.remove('CPIAUCSL')
    for i, v in enumerate(groups):
        if v == 0:
            group0.append(driven_datas[:,i])
            df_list_group0.append(df_list[i])
        elif v == 1:
            group1.append(driven_datas[:, i])
            df_list_group1.append(df_list[i])
        elif v == 2:
            group2.append(driven_datas[:, i])
            df_list_group2.append(df_list[i])
        elif v == 3:
            group3.append(driven_datas[:, i])
            df_list_group3.append(df_list[i])
        elif v == 4:
            group4.append(driven_datas[:, i])
            df_list_group4.append(df_list[i])
        elif v == 5:
            group5.append(driven_datas[:, i])
            df_list_group5.append(df_list[i])
        elif v == 6:
            group6.append(driven_datas[:, i])
            df_list_group6.append(df_list[i])
        elif v == 7:
            group7.append(driven_datas[:, i])
            df_list_group7.append(df_list[i])
    driven_datas_group = np.stack((group0+group1+group2+group3+group4+group5+group6+group7),-1)
    df_list_group = df_list_group0+df_list_group1+df_list_group2+df_list_group3+df_list_group4+df_list_group5+df_list_group6+df_list_group7
    # print(df_list_group)
    print(len(df_list_group0),len(df_list_group1),len(df_list_group2),len(df_list_group3),len(df_list_group4),len(df_list_group5),len(df_list_group6),len(df_list_group7))
    return driven_datas_group, target_datas


class Data_Set(torch_data.Dataset):
    def __init__(self, driven_datas, target_datas, data_index, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.driven_datas = driven_datas
        self.target_datas = target_datas
        self.data_index = data_index

    def __getitem__(self, index):
        i = self.data_index[index]
        driven_datas = th.tensor(self.driven_datas[i:i+self.seq_len,:])
        target_datas = th.tensor(self.target_datas[i:i+self.seq_len,:])
        target_datas_label = th.tensor(self.target_datas[i+self.seq_len:i+self.seq_len+self.pred_len,:])
        # time_feature_datas = th.tensor(self.time_features[i:i+self.seq_len+self.pred_len,:])

        return driven_datas.float(), target_datas.float(), target_datas_label.float()

    def __len__(self):
        return len(self.data_index)

