'''
@Time    : 2020/5/26 21:45
@Author  : sh_lord
@FileName: GBM_MODEL.py

'''
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

class GBM_MODEL:
    def __init__(self):
        self.user_train = "train_preliminary/user.sample.csv"
        self.ad_train = "train_preliminary/ad.sample.csv"
        self.click_log_train = "train_preliminary/click_log.sample.csv"


    def train(self):
        target = 'Disbursed'
        IDcol = 'user_id'
        # Choose all predictors except target & IDcols
        predictors = [x for x in self.user_train.columns if x not in [target, IDcol]]
        gbm0 = GradientBoostingClassifier(random_state=10)
        # 多分类：
        # https://zhuanlan.zhihu.com/p/91652813?utm_source=wechat_session
        # https://www.2cto.com/kf/201802/717234.html
        # https://www.jianshu.com/p/516f009c0875
        modelfit(gbm0, train, test, predictors, printOOB=False)