'''
@Time    : 2020/5/26 23:26
@Author  : sh_lord
@FileName: data_preparation.py

'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

FIELD_SEP = '_'
UNKNOWN = 'UNKNOWN'
LABELENCODER_FILENAME = "GBM/model/label_encoder.pkl"


def getAgeAndGenderFromLabelDict(self, label: str):
    _ = label.split(FIELD_SEP)
    return int(_[-1], int(_[1]))


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
        self.label_dict = self.gen_label_dict()
        # self.df_train = self.data_process()

    def gen_label_dict(self):
        label_dict = {}
        label = 0
        for age in range(1, 11):
            for gender in range(1, 3):
                label_dict[str(age) + FIELD_SEP + str(gender)] = label
                label += 1
        return label_dict



    def add_tf_idf(self):
        pass

    # 用户行为建模, 暂时准备用rnn+crf建模
    def add_user_action_feats(self):
        pass

    def data_process(self):
        df_train = self.user_train.join(self.click_log_train.set_index('user_id'), on='user_id', how='inner')
        df_train = df_train.join(self.ad_train.set_index('creative_id'), on='creative_id', how='inner')
        print(df_train.head(5))
        print(df_train.dtypes)
        # 合并age、 gender
        df_train['label'] = (df_train['age'].map(str) + FIELD_SEP + df_train['gender'].map(str)).map(self.label_dict.get)
        df_train.drop(['age', 'gender'], axis=1, inplace=True)
        print("label:\t", df_train['label'].values)
        var_to_encode = ['product_id', 'industry']
        # Numerical Coding:
        le = LabelEncoder()
        for col in var_to_encode:
            df_train[col] = le.fit_transform(df_train[col])

        # 保存pickle模型
        with open(LABELENCODER_FILENAME, 'wb') as f_pkl:
            pickle.dump(le, f_pkl)

        # One-Hot Coding
        df_train = pd.get_dummies(df_train, columns=var_to_encode)
        for idx in df_train.columns:
            print(idx, end=", ")
        # print(df_train.columns)
        df_train.to_csv('train_modified.csv', index=False)
        return df_train

    def test_data_process(self):
        print("开始读取数据")
        df_ad_test = pd.read_csv("train_preliminary/test/ad.csv")
        df_click_log_test = pd.read_csv("train_preliminary/test/click_log.csv")
        df_test = df_click_log_test.merge(df_ad_test, left_on='creative_id',right_on='creative_id')
        print("merge finished")

        var_to_encode = ['product_id', 'industry']
        # Numerical Coding:
        # 恢复encoder的pickle模型
        with open(LABELENCODER_FILENAME, 'rb') as f_pkl:
            le = pickle.load(f_pkl)
        print("encoder 恢复完成")
        for col in var_to_encode:
            df_test[col] = le.transform(df_test[col])

        # One-Hot Coding
        df_test = pd.get_dummies(df_test, columns=var_to_encode)
        print("特征转化完成")
        df_test.to_csv("test_modified.csv", index=False)

        return df_test





if __name__ == '__main__':
    dp = Data_Preparation()
    dp.data_process()
