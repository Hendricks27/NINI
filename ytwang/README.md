Week1
---
## Done
---
- baseline1: `Week1.1.py`
  - > 参考： kaggle/[Probability of Earthquake: EDA, FE, +5 Models](https://www.kaggle.com/mjbahmani/probability-of-earthquake-eda-fe-5-models/notebook)    
  - 特征工程：
    - 以`150000`大小的窗口对`629145480`个原始数据做处理，对于每个窗口中的数据，取`std`, `kurt`, `max`, `min`, `skew`, `ave`, `sum` 七个特征， 而对应的label为这个窗口最后一个数的`time_tofailure`, 这样就产生了4194个数据的训练集。

  - 模型求解
    - 使用了`SVM`, `LightGBM`, `CatBoost`, `RandomForest`四个模型回归求解
  
  - 模型融合
    - `blending` (简单平均)
- 提交成绩：1.76左右
--- 
- baseline2: `Week1.2.py`
  - > 参考： kaggle/[Earthquakes FE. More features and samples](https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples)
  
  - 特征工程：
    - 在1.1的基础上，增加了131个特征，实际用了138个特征
    - **由于特征工程耗费的时间较长，我把处理好的X_train和X_test放在了data文件夹中**
  - 模型求解
    - 使用了`SVM`, `LightGBM`, `CatBoost`, `RandomForest`，`XGBoost`, `RidgeRegression`六个模型回归求解
  
  - 模型融合
    - `blending` (简单平均)
- 提交成绩：1.65左右

---
## TODO
- 将滑动窗口的移动步长减小，增大数据集
- 增加fft特征
- 对模型调参
- 深度学习模型？