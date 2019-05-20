import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

class FE_throughput():
    def __init__(self, file="./OUTPUT"):
        self.OUTPUT = file
        if(not os.path.exists(file)):
            os.mkdir(file)

    def scale_train_and_test(self, X_train, X_test):
        print("StandardScaler begin...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        X_train_scaled.to_csv(os.path.join(self.OUTPUT, "X_train_scaled.csv"))
        X_test_scaled.to_csv(os.path.join(self.OUTPUT, "X_test_scaled.csv"))
        print("StandardScaler ends.")
        return X_train_scaled, X_test_scaled

    def perason_drop(self, X_train, X_test):
        print("Perason_drop begin...")
        corr_matrix = X_train.corr()
        corr_matrix = corr_matrix.abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        X_train_perason = X_train.drop(to_drop, axis=1)
        X_test_perason = X_test.drop(to_drop, axis=1)
        print(f'{X_train_perason.shape[0]} samples in X_train_perason data and {X_train_perason.shape[1]} columns.')
        print(f'{X_test_perason.shape[0]} samples in X_test_perason data and {X_test_perason.shape[1]} columns.')
        X_train_perason.to_csv(os.path.join(self.OUTPUT, "X_train_perason.csv"))
        X_test_perason.to_csv(os.path.join(self.OUTPUT, "X_test_perason.csv"))
        print("Perason_drop end.")
        return X_train_perason, X_test_perason

    def rf_drop(self, X_train, Y_train, X_test, n_estiamtios_num=10):
        print("RandomForestRegressor begin...")
        rf = RandomForestRegressor(n_estimators=n_estiamtios_num)
        rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_absolute_error', verbose=0,
                      n_jobs=-1)  # 4-fold cross-validation with mae
        rfecv = rfecv.fit(X_train, Y_train.values.ravel())
        print('Optimal number of features :', rfecv.n_features_)
        print('Best features :', X_train.columns[rfecv.support_])

        X_train_rfDrop = X_train[X_train.columns[rfecv.support_].values]
        X_test_rfDrop = X_test[X_test.columns[rfecv.support_].values]
        print(f'{X_train_rfDrop.shape[0]} samples in X_train_rfDrop data and {X_train_rfDrop.shape[1]} columns.')
        print(f'{X_test_rfDrop.shape[0]} samples in X_test_rfDrop data and {X_test_rfDrop.shape[1]} columns.')
        X_train_rfDrop.to_csv(os.path.join(self.OUTPUT, "X_train_rfDrop.csv"))
        X_test_rfDrop.to_csv(os.path.join(self.OUTPUT, "X_test_rfDrop.csv"))
        print("RandomForestRegressor end.")
        return X_train_rfDrop, X_test_rfDrop



if __name__=="__main__":
    temp_test = FE_throughput()

    X_TRAIN_DATA_PATH = "./X_train.csv"
    Y_TRAIN_DATA_PATH = "./y_train.csv"
    X_TEST_DATA_PATH = "./X_test.csv"

    x_train = pd.read_csv(X_TRAIN_DATA_PATH, index_col=0)
    print(f'{x_train.shape[0]} samples in x_train data and {x_train.shape[1]} columns.')
    y_train = pd.read_csv(Y_TRAIN_DATA_PATH, index_col=0)
    print(f'{y_train.shape[0]} samples in y_train data and {y_train.shape[1]} columns.')
    x_test = pd.read_csv(X_TEST_DATA_PATH, index_col="seg_id")
    print(f'{x_test.shape[0]} samples in y_train data and {x_test.shape[1]} columns.')

    temp_test.scale_train_and_test(x_train, x_test)
    temp_test.perason_drop(x_train, x_test)
    temp_test.rf_drop(x_train, y_train, x_test)

    # join_two_train(self, x_train_1, x_train_2)
    # pd.concat([x1, x2], axis=1)

    print("just a test")
