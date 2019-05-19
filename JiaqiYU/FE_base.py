import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from scipy.stats import skew
from sklearn.svm import NuSVR
from scipy.stats import kurtosis
from catboost import CatBoostRegressor,Pool
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

ROOT_PATH = 'input/'
TRAIN_DATA_PATH = ROOT_PATH
TEST_DATA_PATH = ROOT_PATH + 'test/'
SUBMISSION_DATA_PATH = ROOT_PATH

FEATURE_DATA_LENGTH = 150000

class FE_Base(object):
    '''
    This is the base class for feature engineering process
    '''
    feature_data_length = FEATURE_DATA_LENGTH
    test_data_path = TEST_DATA_PATH

    print ('loading data ... ')
    train_data_overall_df = pd.read_csv(TRAIN_DATA_PATH + 'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    sample_submission_df = pd.read_csv(SUBMISSION_DATA_PATH + 'sample_submission.csv', index_col='seg_id')
    segment_number = int(np.floor(train_data_overall_df.shape[0] / FEATURE_DATA_LENGTH)) # 4194

    print ('loading data complete.')

    def __init__(self):
        pass




if __name__=="__main__":

    print()











