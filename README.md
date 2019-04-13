## Week3 本周继续调了random forests，最好的提交成绩 1.572
##        lgb,最好成绩1.567 

### 文件说明 lgb
--- 
- baseline1: `week 3_lgb.ipynb` （4194*138）
- 提交成绩：1.567左右
--- 



## Week2 本周只调了random forests
参考调参方法： https://blog.csdn.net/yanyanyufei96/article/details/71213351
### 文件说明
--- 
- baseline1: `week 2.ipynb` （4194*138）
- 提交成绩：2.08左右
--- 
--- 
- baseline2: `Week2.1.ipynb 
用扩大10倍的数据（41940*138）
- 提交成绩：2.08左右
---
--- 
- baseline3: `Week2.2.ipynb` 
计算了每个feature的VIF,删除了VIF大于10的，最终还剩9个变量
- 提交成绩：2.54左右


### TODO
---
- 继续调lightGBM,catGBM的参数，计算混合模型的mae
--- 
