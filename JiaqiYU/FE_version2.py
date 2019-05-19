import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import skew
from scipy.signal import hann
from scipy.stats import kurtosis
from scipy.signal import hilbert
from scipy.signal import convolve


from sklearn.linear_model import LinearRegression

from JiaqiYU.FE_base import FE_Base

class FE_version2(FE_Base):

    def __init__(self):
        super().__init__()

        self.X_train = pd.DataFrame(index=range(self.segment_number), dtype=np.float64)
        self.Y_train = pd.DataFrame(index=range(self.segment_number), dtype=np.float64, columns=['time_to_failure'])
        self.X_test = pd.DataFrame(columns=self.X_train.columns, dtype=np.float64, index=self.sample_submission_df.index)

    def calc_change_rate(self, x):
        change = (np.diff(x) / x[:-1]).values
        change = change[np.nonzero(change)[0]]
        change = change[~np.isnan(change)]
        change = change[change != -np.inf]
        change = change[change != np.inf]
        return np.mean(change)

    def classic_sta_lta(self, x, length_sta, length_lta):
        sta = np.cumsum(x ** 2)
        sta = np.require(sta, dtype=np.float)
        lta = sta.copy()

        # Compute the STA and the LTA
        sta[length_sta:] = (sta[length_sta:] - sta[:-length_sta]) / length_sta
        lta[length_lta:] = (lta[length_lta:] - lta[:-length_lta]) / length_lta
        sta[:length_lta - 1] = 0 # Pad zeros

        # Avoid division by zero by setting zero values to tiny float
        dtiny = np.finfo(0.0).tiny
        idx = lta < dtiny
        lta[idx] = dtiny

        return sta / lta

    def add_trend_feature(self, arr, abs_values=False):
        idx = np.array(range(len(arr)))
        if abs_values:
            arr = np.abs(arr)
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)
        return lr.coef_[0]

    def generate_features(self, length_of_segment, factor_df, result_df):

        print("Feature Engineering Process Starts.")
        data_slice = self.FEATURE_DATA_LENGTH

        for segment in range(length_of_segment):

            if length_of_segment == len(self.X_train):
                seg = self.train_data_overall_df.iloc[int(segment * data_slice): int(segment * data_slice + data_slice)]
                x = pd.Series(seg['acoustic_data'].values)
                y = seg['time_to_failure'].values[-1]
                result_df.loc[segment, 'time_to_failure'] = y  # y_train

            else:
                seg_id = self.X_test.index.tolist()[segment]
                seg = pd.read_csv('./test/' + seg_id + '.csv')
                segment = seg_id
                x = pd.Series(seg['acoustic_data'].values)

            factor_df.loc[segment, 'ave'] = x.mean()
            factor_df.loc[segment, 'std'] = x.std()
            factor_df.loc[segment, 'max'] = x.max()
            factor_df.loc[segment, 'min'] = x.min()
            factor_df.loc[segment, 'sum'] = x.sum()
            factor_df.loc[segment, 'skew'] = skew(x)
            factor_df.loc[segment, 'kurt'] = kurtosis(x)
            factor_df.loc[segment, 'mad'] = x.mad()
            factor_df.loc[segment, 'med'] = x.median()

            factor_df.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
            factor_df.loc[segment, 'mean_change_rate'] = self.calc_change_rate(x)
            factor_df.loc[segment, 'abs_max'] = np.abs(x).max()
            factor_df.loc[segment, 'abs_min'] = np.abs(x).min()

            factor_df.loc[segment, 'std_first_50000'] = x[:50000].std()
            factor_df.loc[segment, 'std_last_50000'] = x[-50000:].std()
            factor_df.loc[segment, 'std_first_10000'] = x[:10000].std()
            factor_df.loc[segment, 'std_last_10000'] = x[-10000:].std()

            factor_df.loc[segment, 'avg_first_50000'] = x[:50000].mean()
            factor_df.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
            factor_df.loc[segment, 'avg_first_10000'] = x[:10000].mean()
            factor_df.loc[segment, 'avg_last_10000'] = x[-10000:].mean()

            factor_df.loc[segment, 'min_first_50000'] = x[:50000].min()
            factor_df.loc[segment, 'min_last_50000'] = x[-50000:].min()
            factor_df.loc[segment, 'min_first_10000'] = x[:10000].min()
            factor_df.loc[segment, 'min_last_10000'] = x[-10000:].min()

            factor_df.loc[segment, 'max_first_50000'] = x[:50000].max()
            factor_df.loc[segment, 'max_last_50000'] = x[-50000:].max()
            factor_df.loc[segment, 'max_first_10000'] = x[:10000].max()
            factor_df.loc[segment, 'max_last_10000'] = x[-10000:].max()

            factor_df.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())
            factor_df.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())
            factor_df.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])

            factor_df.loc[segment, 'mean_change_rate_first_50000'] = self.calc_change_rate(x[:50000])
            factor_df.loc[segment, 'mean_change_rate_last_50000'] = self.calc_change_rate(x[-50000:])
            factor_df.loc[segment, 'mean_change_rate_first_10000'] = self.calc_change_rate(x[:10000])
            factor_df.loc[segment, 'mean_change_rate_last_10000'] = self.calc_change_rate(x[-10000:])

            factor_df.loc[segment, 'q95'] = np.quantile(x, 0.95)
            factor_df.loc[segment, 'q99'] = np.quantile(x, 0.99)
            factor_df.loc[segment, 'q05'] = np.quantile(x, 0.05)
            factor_df.loc[segment, 'q01'] = np.quantile(x, 0.01)

            factor_df.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
            factor_df.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
            factor_df.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
            factor_df.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

            factor_df.loc[segment, 'trend'] = self.add_trend_feature(x)
            factor_df.loc[segment, 'abs_trend'] = self.add_trend_feature(x, abs_values=True)
            factor_df.loc[segment, 'abs_mean'] = np.abs(x).mean()
            factor_df.loc[segment, 'abs_std'] = np.abs(x).std()

            # signal processing
            factor_df.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
            factor_df.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
            factor_df.loc[segment, 'classic_sta_lta1_mean'] = self.classic_sta_lta(x, 500, 10000).mean()
            factor_df.loc[segment, 'classic_sta_lta2_mean'] = self.classic_sta_lta(x, 5000, 100000).mean()
            factor_df.loc[segment, 'classic_sta_lta3_mean'] = self.classic_sta_lta(x, 3333, 6666).mean()
            factor_df.loc[segment, 'classic_sta_lta4_mean'] = self.classic_sta_lta(x, 10000, 25000).mean()
            factor_df.loc[segment, 'classic_sta_lta5_mean'] = self.classic_sta_lta(x, 50, 1000).mean()
            factor_df.loc[segment, 'classic_sta_lta6_mean'] = self.classic_sta_lta(x, 100, 5000).mean()
            factor_df.loc[segment, 'classic_sta_lta7_mean'] = self.classic_sta_lta(x, 333, 666).mean()
            factor_df.loc[segment, 'classic_sta_lta8_mean'] = self.classic_sta_lta(x, 4000, 10000).mean()
            factor_df.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
            ewma = pd.Series.ewm
            factor_df.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
            factor_df.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
            factor_df.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
            no_of_std = 3
            factor_df.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
            factor_df.loc[segment, 'MA_700MA_BB_high_mean'] = (factor_df.loc[segment, 'Moving_average_700_mean'] + no_of_std * factor_df.loc[segment, 'MA_700MA_std_mean']).mean()
            factor_df.loc[segment, 'MA_700MA_BB_low_mean'] = (factor_df.loc[segment, 'Moving_average_700_mean'] - no_of_std * factor_df.loc[segment, 'MA_700MA_std_mean']).mean()
            factor_df.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
            factor_df.loc[segment, 'MA_400MA_BB_high_mean'] = (factor_df.loc[segment, 'Moving_average_700_mean'] + no_of_std * factor_df.loc[segment, 'MA_400MA_std_mean']).mean()
            factor_df.loc[segment, 'MA_400MA_BB_low_mean'] = (factor_df.loc[segment, 'Moving_average_700_mean'] - no_of_std * factor_df.loc[segment, 'MA_400MA_std_mean']).mean()
            factor_df.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
            factor_df.drop('Moving_average_700_mean', axis=1, inplace=True)

            factor_df.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
            factor_df.loc[segment, 'q999'] = np.quantile(x, 0.999)
            factor_df.loc[segment, 'q001'] = np.quantile(x, 0.001)
            factor_df.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)

            for windows in [10, 100, 1000]:
                x_roll_std = x.rolling(windows).std().dropna().values
                x_roll_mean = x.rolling(windows).mean().dropna().values

                factor_df.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
                factor_df.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
                factor_df.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
                factor_df.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
                factor_df.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
                factor_df.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
                factor_df.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
                factor_df.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
                factor_df.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
                factor_df.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
                factor_df.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

                factor_df.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
                factor_df.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
                factor_df.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
                factor_df.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
                factor_df.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
                factor_df.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
                factor_df.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
                factor_df.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
                factor_df.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
                factor_df.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
                factor_df.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

        print("Feature Engineering Process Ends.")
        return factor_df, result_df