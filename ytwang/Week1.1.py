from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from eli5.sklearn import PermutationImportance
from catboost import CatBoostRegressor,Pool
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import norm
from scipy import linalg
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import graphviz
import warnings
import random
import eli5
import shap  # package used to calculate Shap values
import time
import glob
import sys
import os


# Reference: https://www.kaggle.com/mjbahmani/probability-of-earthquake-eda-fe-5-models/notebook
print("Reading csv files...")
train = pd.read_csv('./train.csv' , dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print("Train: rows:{} columns:{}".format(train.shape[0], train.shape[1]))

rows = 150_000
segments = int(np.floor(train.shape[0] / rows)) # 4194

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min','sum','skew','kurt']) # 使用对应的特征
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])
submission = pd.read_csv('./sample_submission.csv', index_col='seg_id')
print("Reading ends.")

# 特征工程， 建立y_train, X_train, X_test
# 特征重要性排名：std, kurt, max, min, skew, ave, sum
print("Feature engineering...")
for segment in range(segments):
    # y_train and X_train
    seg = train.iloc[segment * rows:segment * rows + rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]  # 只取倒数第一个值，只过了0.0375s

    y_train.loc[segment, 'time_to_failure'] = y  # y_train
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[segment, 'sum'] = x.sum()
    X_train.loc[segment, 'skew'] = skew(x)
    X_train.loc[segment, 'kurt'] = kurtosis(x)

X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
for seg_id in X_test.index:
    # X_test
    seg = pd.read_csv('./test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'sum'] = x.sum()
    X_test.loc[seg_id, 'skew'] = skew(x)
    X_test.loc[seg_id, 'kurt'] = kurtosis(x)

X = X_train.copy()
y = y_train.copy()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 正则化X
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature engineering ends.")


print("SVM...")
svm = NuSVR()
svm.fit(X_train_scaled, y_train.values.flatten())
y_pred_svm = svm.predict(X_train_scaled)
score = mean_absolute_error(y_train.values.flatten(), y_pred_svm)
print(f'Score: {score:0.3f}')
y_pred_svm= svm.predict(X_test_scaled)
submission['time_to_failure'] = y_pred_svm
submission.to_csv('submission_svm.csv')
print("SVM ends.")



print("LightGBM...")
folds = KFold(n_splits=5, shuffle=True, random_state=42)
params = {'objective' : "regression",
               'boosting':"gbdt",
               'metric':"mae",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.001,
               'num_leaves' : 52,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.85,
               'bagging_freq' : 1,
               'bagging_fraction' : 0.85,
               'min_data_in_leaf' : 10,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : -1}
y_pred_lgb = np.zeros(len(X_test_scaled))
for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(X))):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    model = lgb.LGBMRegressor(**params, n_estimators=22000, n_jobs=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
              verbose=1000, early_stopping_rounds=200)

    y_pred_valid = model.predict(X_valid)
    y_pred_lgb += model.predict(X_test_scaled, num_iteration=model.best_iteration_) / folds.n_splits
submission['time_to_failure'] = y_pred_lgb
submission.to_csv('submission_lgb.csv')
print("LightGBM ends.")

# CatBoost
print("CatBoost...")
train_pool = Pool(X,y)
cat_model = CatBoostRegressor(
                               iterations=3000,  # change 25 to 3000 to get best performance
                               learning_rate=0.03,
                               eval_metric='MAE',
                              )
cat_model.fit(X,y,silent=True)
y_pred_cat = cat_model.predict(X_test)
submission['time_to_failure'] = y_pred_cat
submission.to_csv('submission_cat.csv')
print("CatBoost ends.")

# RandomForest
print("RandomForest...")
rfc_model = RandomForestRegressor(random_state=0).fit(X, y.values.flatten())
y_pred_rf=rfc_model.predict(X_test)
submission['time_to_failure'] = y_pred_rf
submission.to_csv('submission_rf.csv')
print("RandomForest Ends.")


# blending
blending = y_pred_svm*0.5 + y_pred_lgb*0.5
submission['time_to_failure'] = blending
submission.to_csv('submission_lgb_svm.csv')

blending = y_pred_svm*0.5 + y_pred_cat*0.5
submission['time_to_failure'] = blending
submission.to_csv('submission_cat_svm.csv')
