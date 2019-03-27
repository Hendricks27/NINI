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
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Reference: https://www.kaggle.com/mjbahmani/probability-of-earthquake-eda-fe-5-models/notebook
# Reference: https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
print("Reading csv files...")
train = pd.read_csv('./train.csv' , dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print("Train: rows:{} columns:{}".format(train.shape[0], train.shape[1]))

rows = 150_000
segments = int(np.floor(train.shape[0] / rows)) # 4194

X_train = pd.DataFrame(index=range(segments), dtype=np.float64) # 使用对应的特征
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])
submission = pd.read_csv('./sample_submission.csv', index_col='seg_id')
print("Reading ends.")


print("Feature engineering...")
# utils
def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]  # 因为会返回tuple，所以要加[0]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


# y = ax + b, solve a
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


print("Feature engineering...")
scale = 1  # test:使滑动窗口缩小10倍
print("y_train and X_train engineering...")
for segment in range(segments):
    # y_train and X_train
    #     if segment * rows / scale + rows > train.shape[0]:
    #         break
    seg = train.iloc[int(segment * rows / scale): int(segment * rows / scale + rows)]
    x = pd.Series(seg['acoustic_data'].values)
    y = seg['time_to_failure'].values[-1]  # 只取倒数第一个值，只过了0.0375s, 如果遇到阶跃怎么办？

    y_train.loc[segment, 'time_to_failure'] = y  # y_train
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[segment, 'sum'] = x.sum()
    X_train.loc[segment, 'skew'] = skew(x)
    X_train.loc[segment, 'kurt'] = kurtosis(x)
    X_train.loc[segment, 'mad'] = x.mad()
    X_train.loc[segment, 'med'] = x.median()

    X_train.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
    X_train.loc[segment, 'mean_change_rate'] = calc_change_rate(x)
    X_train.loc[segment, 'abs_max'] = np.abs(x).max()
    X_train.loc[segment, 'abs_min'] = np.abs(x).min()

    X_train.loc[segment, 'std_first_50000'] = x[:50000].std()
    X_train.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X_train.loc[segment, 'std_first_10000'] = x[:10000].std()
    X_train.loc[segment, 'std_last_10000'] = x[-10000:].std()

    X_train.loc[segment, 'avg_first_50000'] = x[:50000].mean()
    X_train.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
    X_train.loc[segment, 'avg_first_10000'] = x[:10000].mean()
    X_train.loc[segment, 'avg_last_10000'] = x[-10000:].mean()

    X_train.loc[segment, 'min_first_50000'] = x[:50000].min()
    X_train.loc[segment, 'min_last_50000'] = x[-50000:].min()
    X_train.loc[segment, 'min_first_10000'] = x[:10000].min()
    X_train.loc[segment, 'min_last_10000'] = x[-10000:].min()

    X_train.loc[segment, 'max_first_50000'] = x[:50000].max()
    X_train.loc[segment, 'max_last_50000'] = x[-50000:].max()
    X_train.loc[segment, 'max_first_10000'] = x[:10000].max()
    X_train.loc[segment, 'max_last_10000'] = x[-10000:].max()

    X_train.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())
    X_train.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())
    X_train.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])

    X_train.loc[segment, 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
    X_train.loc[segment, 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
    X_train.loc[segment, 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
    X_train.loc[segment, 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    X_train.loc[segment, 'q95'] = np.quantile(x, 0.95)
    X_train.loc[segment, 'q99'] = np.quantile(x, 0.99)
    X_train.loc[segment, 'q05'] = np.quantile(x, 0.05)
    X_train.loc[segment, 'q01'] = np.quantile(x, 0.01)

    X_train.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
    X_train.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
    X_train.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
    X_train.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

    X_train.loc[segment, 'trend'] = add_trend_feature(x)
    X_train.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
    X_train.loc[segment, 'abs_mean'] = np.abs(x).mean()
    X_train.loc[segment, 'abs_std'] = np.abs(x).std()

    # signal processing
    X_train.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    X_train.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    X_train.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    X_train.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    X_train.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    X_train.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    X_train.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
    X_train.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
    X_train.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
    X_train.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
    X_train.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    X_train.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    X_train.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    X_train.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
    no_of_std = 3
    X_train.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    X_train.loc[segment, 'MA_700MA_BB_high_mean'] = (
                X_train.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_train.loc[
            segment, 'MA_700MA_std_mean']).mean()
    X_train.loc[segment, 'MA_700MA_BB_low_mean'] = (
                X_train.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_train.loc[
            segment, 'MA_700MA_std_mean']).mean()
    X_train.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    X_train.loc[segment, 'MA_400MA_BB_high_mean'] = (
                X_train.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_train.loc[
            segment, 'MA_400MA_std_mean']).mean()
    X_train.loc[segment, 'MA_400MA_BB_low_mean'] = (
                X_train.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_train.loc[
            segment, 'MA_400MA_std_mean']).mean()
    X_train.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    X_train.drop('Moving_average_700_mean', axis=1, inplace=True)

    X_train.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
    X_train.loc[segment, 'q999'] = np.quantile(x, 0.999)
    X_train.loc[segment, 'q001'] = np.quantile(x, 0.001)
    X_train.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values

        X_train.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X_train.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X_train.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X_train.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X_train.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X_train.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X_train.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X_train.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X_train.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X_train.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X_train.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X_train.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X_train.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X_train.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X_train.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X_train.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X_train.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X_train.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X_train.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X_train.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X_train.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X_train.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
print("y_train and X_train engineering ends.")


print("X_test engineering...")
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
for seg_id in X_test.index:
    # X_test
    seg = pd.read_csv('./test/' + seg_id + '.csv')
    segment = seg_id
    x = pd.Series(seg['acoustic_data'].values)

    X_test.loc[segment, 'ave'] = x.mean()
    X_test.loc[segment, 'std'] = x.std()
    X_test.loc[segment, 'max'] = x.max()
    X_test.loc[segment, 'min'] = x.min()
    X_test.loc[segment, 'sum'] = x.sum()
    X_test.loc[segment, 'skew'] = skew(x)
    X_test.loc[segment, 'kurt'] = kurtosis(x)
    X_test.loc[segment, 'mad'] = x.mad()
    X_test.loc[segment, 'med'] = x.median()

    X_test.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
    X_test.loc[segment, 'mean_change_rate'] = calc_change_rate(x)
    X_test.loc[segment, 'abs_max'] = np.abs(x).max()
    X_test.loc[segment, 'abs_min'] = np.abs(x).min()

    X_test.loc[segment, 'std_first_50000'] = x[:50000].std()
    X_test.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X_test.loc[segment, 'std_first_10000'] = x[:10000].std()
    X_test.loc[segment, 'std_last_10000'] = x[-10000:].std()

    X_test.loc[segment, 'avg_first_50000'] = x[:50000].mean()
    X_test.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
    X_test.loc[segment, 'avg_first_10000'] = x[:10000].mean()
    X_test.loc[segment, 'avg_last_10000'] = x[-10000:].mean()

    X_test.loc[segment, 'min_first_50000'] = x[:50000].min()
    X_test.loc[segment, 'min_last_50000'] = x[-50000:].min()
    X_test.loc[segment, 'min_first_10000'] = x[:10000].min()
    X_test.loc[segment, 'min_last_10000'] = x[-10000:].min()

    X_test.loc[segment, 'max_first_50000'] = x[:50000].max()
    X_test.loc[segment, 'max_last_50000'] = x[-50000:].max()
    X_test.loc[segment, 'max_first_10000'] = x[:10000].max()
    X_test.loc[segment, 'max_last_10000'] = x[-10000:].max()

    X_test.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())
    X_test.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())
    X_test.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])

    X_test.loc[segment, 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
    X_test.loc[segment, 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
    X_test.loc[segment, 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
    X_test.loc[segment, 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    X_test.loc[segment, 'q95'] = np.quantile(x, 0.95)
    X_test.loc[segment, 'q99'] = np.quantile(x, 0.99)
    X_test.loc[segment, 'q05'] = np.quantile(x, 0.05)
    X_test.loc[segment, 'q01'] = np.quantile(x, 0.01)

    X_test.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
    X_test.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
    X_test.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
    X_test.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

    X_test.loc[segment, 'trend'] = add_trend_feature(x)
    X_test.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
    X_test.loc[segment, 'abs_mean'] = np.abs(x).mean()
    X_test.loc[segment, 'abs_std'] = np.abs(x).std()

    # signal processing
    X_test.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    X_test.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    X_test.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    X_test.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    X_test.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    X_test.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    X_test.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
    X_test.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
    X_test.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
    X_test.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
    X_test.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    X_test.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    X_test.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    X_test.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
    no_of_std = 3
    X_test.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    X_test.loc[segment, 'MA_700MA_BB_high_mean'] = (
                X_test.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_test.loc[
            segment, 'MA_700MA_std_mean']).mean()
    X_test.loc[segment, 'MA_700MA_BB_low_mean'] = (
                X_test.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_test.loc[
            segment, 'MA_700MA_std_mean']).mean()
    X_test.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    X_test.loc[segment, 'MA_400MA_BB_high_mean'] = (
                X_test.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_test.loc[
            segment, 'MA_400MA_std_mean']).mean()
    X_test.loc[segment, 'MA_400MA_BB_low_mean'] = (
                X_test.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_test.loc[
            segment, 'MA_400MA_std_mean']).mean()
    X_test.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    X_test.drop('Moving_average_700_mean', axis=1, inplace=True)

    X_test.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
    X_test.loc[segment, 'q999'] = np.quantile(x, 0.999)
    X_test.loc[segment, 'q001'] = np.quantile(x, 0.001)
    X_test.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values

        X_test.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X_test.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X_test.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X_test.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X_test.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X_test.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X_test.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X_test.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X_test.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X_test.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X_test.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X_test.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X_test.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X_test.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X_test.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X_test.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X_test.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X_test.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X_test.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X_test.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X_test.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X_test.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
print("X_test engineering ends.")


# 正则化
print("FillNA and standardScalar...")
means_dict = {}
for col in X_train.columns:
    if X_train[col].isnull().any():
        print(col)
        mean_value = X_train.loc[X_train[col] != -np.inf, col].mean()
        X_train.loc[X_train[col] == -np.inf, col] = mean_value
        X_train[col] = X_train[col].fillna(mean_value)
        means_dict[col] = mean_value

for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
        X_test[col] = X_test[col].fillna(means_dict[col])

X = X_train.copy()
y = y_train.copy()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
print("FillNA and standardScalar ends.")
print("Feature engineering ends.")


# Building algorithm models
print("Building algorithm models...")
# alogorithm utils
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_train, params=None, folds=folds, model_type='lgb',
                plot_feature_importance=False, model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                      verbose=10000, early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1, )

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');

            return oof, prediction, feature_importance
        return oof, prediction

    else:
        return oof, prediction


# SVM
print("NuSVR...")
model = NuSVR(gamma='scale', nu=0.9, C=10.0, tol=0.01)
oof_svr, prediction_svr = train_model(X=X_train_scaled, X_test=X_test_scaled, params=None, model_type='sklearn', model=model)


model = NuSVR(gamma='scale', nu=0.7, tol=0.01, C=1.0)
oof_svr1, prediction_svr1 = train_model(X=X_train_scaled, X_test=X_test_scaled, params=None, model_type='sklearn', model=model)
print("NuSVR ends.")

# LightGBM
print("LightGBM...")
params = {'num_leaves': 54,
          'min_data_in_leaf': 79,
          'objective': 'huber',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501
         }
oof_lgb, prediction_lgb, feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)
print("LightGBM ends.")


# XGBoost
print("XGBoost...")
xgb_params = {'eta': 0.03,
              'max_depth': 10,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': 4}
oof_xgb, prediction_xgb = train_model(X=X_train_scaled, X_test=X_test_scaled, params=xgb_params, model_type='xgb')
print("XGBoost ends")

# CatBoost
# 设置20000 iters非常耗时，一个fold耗时24min
print("CatBoost...")
params = {'loss_function':'MAE'}
oof_cat, prediction_cat = train_model(X=X_train_scaled, X_test=X_test_scaled, params=params, model_type='cat')
print("CatBoost ends.")


# RandomForest
print("RandomForest...")
rfc_model = RandomForestRegressor(random_state=0).fit(X, y.values.flatten())
y_pred_rf = rfc_model.predict(X_test)
print("RandomForest Ends.")


# KernelRidge
print("KernelRidge...")
model = KernelRidge(kernel='rbf', alpha=0.15, gamma=0.01)
oof_r, prediction_r = train_model(X=X_train_scaled, X_test=X_test_scaled, params=None, model_type='sklearn', model=model)
print("KernelRidge ends.")

# blending
submission['time_to_failure'] = (prediction_lgb + prediction_xgb + prediction_svr + prediction_svr1 + prediction_cat + prediction_r) / 6
print(submission.head())
submission.to_csv('submission_lgb_xgb_svr_cat_r.csv')
print("The End.")