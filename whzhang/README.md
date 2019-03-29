## Week 1  
### Done
Reference: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction/notebook
https://www.kaggle.com/mjbahmani/probability-of-earthquake-eda-fe-5-models/notebook  
and Yummy's work.

####Version1.1 `baseline ver1.1.ipynb`
1. data visualization  
取1%的全数据，画acoustic data和time to failure, 发现在每一个failure（time=0）前，都有acoustic data的一个小高峰，有的是在靠近failure的地方，有的是在两次failure中间，说明可以通过acoustic信号来预测failure time。
2. 增加特征的方法按照王意天1.2版本, in total 138 features  
reference:https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples 
   * Usual aggregations: mean, std, min and max
   * Average difference between the consequitive values in absolute and percent values  
   * Absolute min and max vallues  
   * Aforementioned aggregations for first and last 10000 and 50000 values - I think these data should be useful
   * Max value to min value and their differencem also count of values bigger than 500 (arbitrary threshold);
   * Quantile features
   * Trend features
   * Rolling features

####Version1.2 `week1_CNN.py`
1. Models  
   * 尝试了CNN，参考https://www.kaggle.com/fanconic/earthquake-cnn  
   没有扩增feature
   
   
   
   