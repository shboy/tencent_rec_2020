'''
@Time    : 2020/5/26 23:26
@Author  : sh_lord
@FileName: data_preparation.py

'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Data_Preparation:
    # user: Index(['user_id', 'age', 'gender'], dtype='object'),
    # ad: Index(['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry'], dtype='object'),
    # context: Index(['time', 'user_id', 'creative_id', 'click_times'], dtype='object'))
    def __init__(self):
        self.user_train = pd.read_csv("train_preliminary/user.sample.csv")
        # print(self.user_train.columns)
        self.ad_train = pd.read_csv("train_preliminary/ad.sample.csv")
        # print(self.ad_train.columns)
        self.click_log_train = pd.read_csv("train_preliminary/click_log.sample.csv")
        # print(self.click_log_train.columns)
        self.df_train = self.data_process()

    def data_process(self):
        df_train = self.user_train.join(self.click_log_train.set_index('user_id'), on='user_id', how='inner')
        df_train = df_train.join(self.ad_train.set_index('creative_id'), on='creative_id', how='inner')
        print(df_train.head(5))
        print(df_train.dtypes)
        # Numerical Coding:
        le = LabelEncoder()
        var_to_encode = ['product_id', 'industry']
        for col in var_to_encode:
            df_train[col] = le.fit_transform(df_train[col])
        # One-Hot Coding
        df_train = pd.get_dummies(df_train, columns=var_to_encode)
        for idx in df_train.columns:
            print(idx, ", ")
        # print(df_train.columns)
        df_train.to_csv('train_modified.csv', index=False)
        return df_train

    def train(self):
        target = 'Disbursed'
        IDcol = 'ID'


if __name__ == '__main__':
    dp = Data_Preparation()
    dp.data_process()

