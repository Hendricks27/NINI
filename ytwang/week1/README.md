## Week1
### 文件说明
- baseline1: `Week1.1.py`
  - > 参考： kaggle/[Probability of Earthquake: EDA, FE, +5 Models](https://www.kaggle.com/mjbahmani/probability-of-earthquake-eda-fe-5-models/notebook)    
  - 特征工程：
    - 对于`X_train`: 以`150000`大小的窗口对`629145480`个原始数据做处理，对于每个窗口中的数据，取`std`, `kurt`, `max`, `min`, `skew`, `ave`, `sum` 七个特征
      -  矩阵大小4194*7
    - 对于`y_train`: 取对应窗口最后一个数的`time_to_failure`
      - 矩阵大小4194*1
      - 因为150000的窗口实际只过了0.0375s，所以这么取是合理的，另外测试集也是要输出最后一个数的`time_to_failure`

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
    - 使用了`K-fold`增加模型的泛化能力
  
  - 模型融合
    - `blending` (简单平均)
- 提交成绩：1.65左右
---
- `ReadData.py`
  - 用来把已经跑好的`X_train`, `X_test`, `y_train`读入
---
- `Week1.3.ipynb`
  - 包含了一些模型跑出来的结果
  
### 特征释疑
--- 
  - `skew`
    - 偏度, 统计数据分布非对称程度
    - $E[(\frac{X-\mu}{\sigma})^3]$
  - `kurt`
    - 峰值，峰度（Kurtosis）衡量实数随机变量概率分布的峰态。峰度高就意味着方差增大是由低频度的大于或小于平均值的极端差值引起的。
    - $E[(\frac{X-\mu}{\sigma})^4]$
  - `mad`
    - mean absolute deviation
    - $E[|x-m(X)|]$
  - `Hilbert_mean`
    - > CSDN/[Hilbert-Huang Transform（希尔伯特-黄变换）](https://www.cnblogs.com/hdu-zsk/p/4799470.html)
    - 希尔伯特变换的本质是一个90°相移器
  - `hann_window_mean`
    - x对一个hann_window做卷积后取均值
      - hann
      - $w(n)=0.5-0.5cos(\frac{2\pi*n}{M-1})$
  - `sta/lta`
    - > cnblogs/[STA/LTA方法](http://www.cnblogs.com/seisjun/p/6907229.html)
    - STA是用于捕捉地震信号的时间窗，因此STA越短，就对短周期的地震信号捕捉越有效；LTA是用于衡量时间窗内的平均噪声
  - `exp_Moving_average`
    - > CSDN/[pandas: ewm的参数设置](https://blog.csdn.net/Papageno_Xue/article/details/82705157)
    - 指数滑动窗口

### TODO
---
- 将滑动窗口的移动步长减小，增大数据集
  - 增大了10倍数据集，将分数提高0.1，1.516左右
- 滑动窗口，考虑把一些地震发生时间在数据中段的去掉
- 增加傅里叶变换等特征
- 对模型调参
- 深度学习模型？