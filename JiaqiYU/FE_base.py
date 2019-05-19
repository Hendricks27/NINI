import numpy as np
import pandas as pd
import os


class FE_Base(object):
    '''
    This is the base class for feature engineering process
    '''
    def __init__(self):
        self.ROOT_PATH = r'D:\boundless-knowledge\project\Earthquake\LANL-Earthquake-Prediction\data_orgin'

        self.TRAIN_DATA_PATH = self.ROOT_PATH
        self.TEST_DATA_PATH = os.path.join(self.ROOT_PATH, 'test/')
        self.SUBMISSION_DATA_PATH = self.ROOT_PATH

        self.FEATURE_DATA_LENGTH = 150000
        # feature_data_length = FEATURE_DATA_LENGTH
        # test_data_path = TEST_DATA_PATH

        print('loading data ... ')
        self.train_data_overall_df = pd.read_csv(os.path.join(self.TRAIN_DATA_PATH, 'train.csv'),
                                            dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
        self.sample_submission_df = pd.read_csv(os.path.join(self.SUBMISSION_DATA_PATH, 'sample_submission.csv'),
                                           index_col='seg_id')
        self.segment_number = int(np.floor(self.train_data_overall_df.shape[0] / self.FEATURE_DATA_LENGTH))  # 4194
        print('loading data complete.')

if __name__=="__main__":

    print()











