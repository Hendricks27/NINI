import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

dir = "./data/41940samples/"
X_train = pd.read_csv(dir + "/X_train_41940samples_138features_filtered.csv", index_col=0)
X_test = pd.read_csv(dir + "X_test_41940samples_138features_filtered.csv", index_col="seg_id")
y_train = pd.read_csv(dir + "y_train_41940sasmples.csv",index_col=0)

for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
        X_test[col] = X_test[col].fillna(means_dict[col])

# 正则化X
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print(f'{X_train_scaled.shape[0]} samples in new train data and {X_train_scaled.shape[1]} columns.')
print(X_train_scaled.head())
print(f'{X_test_scaled.shape[0]} samples in new test data and {X_test_scaled.shape[1]} columns.')
print(X_test_scaled.head())
print(f'{y_train.shape[0]} samples in y_train data and {y_train.shape[1]} columns.')
print(y_train.head)