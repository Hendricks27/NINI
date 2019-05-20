import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import os
from FE_base import FE_Base
import librosa, librosa.display

class FE_version1(FE_Base):

    def __init__(self):
        super().__init__()

        self.X_train = pd.DataFrame(index=range(self.segment_number), dtype=np.float64)
        self.Y_train = pd.DataFrame(index=range(self.segment_number), dtype=np.float64, columns=['time_to_failure'])
        self.X_test = pd.DataFrame(dtype=np.float64, index=self.sample_submission_df.index)

    def generate_training_features(self):

        print("Feature engineering for training data...")

        for segment in range(self.segment_number):
            data_slice = self.FEATURE_DATA_LENGTH

            seg = self.train_data_overall_df.iloc[segment * data_slice: segment * data_slice + data_slice]
            x = seg['acoustic_data'].values
            y = seg['time_to_failure'].values[-1]  # 只取倒数第一个值，只过了0.0375s

            self.Y_train.loc[segment, 'time_to_failure'] = y  # y_train
            self.X_train.loc[segment, 'ave'] = x.mean()
            self.X_train.loc[segment, 'std'] = x.std()
            self.X_train.loc[segment, 'max'] = x.max()
            self.X_train.loc[segment, 'min'] = x.min()
            self.X_train.loc[segment, 'sum'] = x.sum()
            self.X_train.loc[segment, 'skew'] = skew(x)
            self.X_train.loc[segment, 'kurt'] = kurtosis(x)

        print("Feature engineering for training data complete")

        return

    def generate_testing_features(self):

        print ("Feature engineering for testing data...")
        for seg_id in self.X_test.index:
            seg = pd.read_csv(os.path.join(self.TEST_DATA_PATH, seg_id + '.csv'))
            x = seg['acoustic_data'].values

            self.X_test.loc[seg_id, 'ave'] = x.mean()
            self.X_test.loc[seg_id, 'std'] = x.std()
            self.X_test.loc[seg_id, 'max'] = x.max()
            self.X_test.loc[seg_id, 'min'] = x.min()
            self.X_test.loc[seg_id, 'sum'] = x.sum()
            self.X_test.loc[seg_id, 'skew'] = skew(x)
            self.X_test.loc[seg_id, 'kurt'] = kurtosis(x)

        print("Feature engineering for testing data complete.")

        return

    def add_mfcc(self):
        print("Feature engineering for adding mfcc...")

        for segment in range(self.segment_number):
            data_slice = self.FEATURE_DATA_LENGTH
            seg = self.train_data_overall_df.iloc[segment * data_slice: segment * data_slice + data_slice]
            x = pd.Series(seg['acoustic_data'].values)
            mfcc = librosa.feature.mfcc(np.array(x).astype('float32'))
            mfcc_mean = mfcc.mean(axis=1)
            for i, each_mfcc_mean in enumerate(mfcc_mean):
                key = 'mfcc_{}'.format(i)
                self.X_train.loc[segment, key] = each_mfcc_mean

        for seg_id in self.X_test.index:
            seg = pd.read_csv(os.path.join(self.TEST_DATA_PATH, seg_id + '.csv'))
            segment = seg_id
            x = pd.Series(seg['acoustic_data'].values)
            mfcc = librosa.feature.mfcc(np.array(x).astype('float32'))
            mfcc_mean = mfcc.mean(axis=1)
            for i, each_mfcc_mean in enumerate(mfcc_mean):
                key = 'mfcc_{}'.format(i)
                self.X_test.loc[segment, key] = each_mfcc_mean

        print("Feature engineering for adding mfcc complete...")


if __name__=="__main__":
    temp_test = FE_version1()
    # temp_test.generate_training_features()
    # temp_test.generate_testing_features()
    temp_test.add_mfcc()
    temp_test.X_train.to_csv("X_train.csv")
    # temp_test.Y_train.to_csv("y_train.csv")
    temp_test.X_test.to_csv("X_test.csv")
    print("just a test")
