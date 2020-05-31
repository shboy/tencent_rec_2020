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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import pickle
import argparse

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

from data_preparation import LABEL_ENCODER_FILENAME, Data_Preparation, getAgeAndGenderFromLabelDict

rcParams['figure.figsize'] = 12, 4

MODEL_FILENAME = "GBM/model/GBM.pb"

class GBM_MODEL:
    def __init__(self):
        # self.user_train = "train_preliminary/user.sample.csv"
        # self.ad_train = "train_preliminary/ad.sample.csv"
        # self.click_log_train = "train_preliminary/click_log.sample.csv"
        self.train_data = pd.read_csv("train_modified.csv")
        # self.test_data = pd.read_csv()

    def modelfit(self, alg, dtrain, dtest, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
        # Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain['label'])

        # Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

        # Perform cross-validation:
        if performCV:
            cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['label'], cv=cv_folds
                                                        ,scoring='accuracy')

        # Print model report:
        print("\nModel Report")

        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions))

        # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob))


        if performCV:
            print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g "
                  "| Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

        # Print Feature Importance:
        if printFeatureImportance:
            feat_imp = pd.Series(alg.feature_importances_, predictors)
            feat_imp = feat_imp[feat_imp>0].sort_values(ascending=False)[:30]
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.savefig("./FeatureImportance.png")
            # plt.show()

        with open(MODEL_FILENAME, 'wb') as f_pkl:
            pickle.dump(alg, f_pkl)



    def train(self):
        target = 'label'
        IDcol = 'user_id'
        # Choose all predictors except target & IDcols
        predictors = [x for x in self.train_data.columns if x not in [target, IDcol]]
        gbm0 = GradientBoostingClassifier(random_state=10)
        # 多分类：
        # https://zhuanlan.zhihu.com/p/91652813?utm_source=wechat_session
        # https://www.2cto.com/kf/201802/717234.html
        # https://www.jianshu.com/p/516f009c0875
        self.modelfit(gbm0, self.train_data, self.train_data, predictors)

    def predict(self):
        dp = Data_Preparation()
        df_test = dp.test_data_process()

        with open(MODEL_FILENAME, 'rb') as f_pkl:
            gbm: GradientBoostingClassifier = pickle.load(f_pkl)

        result = gbm.predict(df_test)
        l_age = []
        l_gender = []
        for label in result:
            age, gender = getAgeAndGenderFromLabelDict(label)
            l_age.append(age)
            l_gender.append(gender)

        df_test['predicted_age'] = l_age
        df_test['predicted_gender'] = l_gender
        df_test_result = df_test[['user_id', 'predicted_age', 'predicted_gender']]
        df_test_result[['predicted_age', 'predicted_gender']].groupby('user_id').agg(lambda x: np.mean(x.mode()[0])).reset_index()
        df_test_result.to_csv('/submission/result.csv', sep=',', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        help='train or predict')
    args = parser.parse_args()

    gm = GBM_MODEL()
    if args.action == "train":
        gm.train()
    elif args.action == "predict":
        gm.predict()
    else:
        raise Exception("action wrong, train or predict")