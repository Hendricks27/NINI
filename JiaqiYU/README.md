## Week 2 - JiaqiYU
### Done
- Model Review: `RNN & LSTM & GRU`
  - > 参考： Course from Hung-yi Lee/[Course Homepage](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html)    
    - Basic framework of RNN, LSTM and GRU
    - Difference between LSTM & GRU
    - The Reason why LSTM models were used
  
--- 
- baseline1: `Week2.1.py`
  - > 参考： kaggle/[RNN starter for huge time series](https://www.kaggle.com/mayer79/rnn-starter-for-huge-time-series)
  
  - 特征工程：
    - 随机从原始数据中找end point,之后往前取150,000个数据点，n_steps=150, step_length=1000
    - 对每个piece，求所有1000个数据点/最后100个数据点/最后10个数据点的以下特征：
    - 特征：均值，标准差，最大值，最小值
    - batch_size = 32
  - 模型求解
    - 使用了`RNN`中的`GRU`模型
    - 第一层`GRU`层，输出48个节点，第二层`Dense`层，输出10个节点，第三层输出层
  
- 成绩：1.516

- baseline2: `Week2.2.py`
  - 特征工程：同上
  - 模型求解
    - 使用了`RNN`中的`LSTM`模型
    - 第一层`LSTM`层，输出48个节点，第二层`Dense`层，输出10个节点，第三层输出层
  
- 成绩：1.526
- `LSTM`和`GRU`的简单平均提交成绩：1.552

- baseline3: `Week2.3.py`
  - 特征工程：
    - 对每个piece，求所有1000个数据点/最后200/100/50/20/10个数据点的特征
  - 模型求解
    - 第一层`GRU`层，输出32个节点，第二层`GRU`层，输出16个节点，第三层`Dense`层，输出8个节点，第三层输出层

- 成绩：？
  
---
### TODO
- 特征工程优化
- 神经网络结构优化
- 调参：n_step, step_length, batch_size etc.
- Training-Validation机制优化
- 添加过拟合机制
