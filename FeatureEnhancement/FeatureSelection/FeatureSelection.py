import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
import pylab as pl

# 使用Titanic数据集，通过特征筛选的方法一步步提升决策树的预测性能
titanic = pd.read_csv('titanic.txt')
# 分离数据特征与预测目标
x = titanic.drop(['row.names', 'name', 'survived'], axis=1)
y = titanic['survived']
x['age'].fillna(x['age'].mean(), inplace=True)
x.fillna('UNKNOWN', inplace=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=30)
vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient='records'))
x_test = vec.transform(x_test.to_dict(orient='records'))
print(len(vec.feature_names_))
# 使用决策树模型依靠所有特征进行预测和评估性能
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test))
# 筛选前20%的特征，使用相同配置的决策树进行预测和评估
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
x_train_fs = fs.fit_transform(x_train, y_train)
dt.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print(dt.score(x_test_fs, y_test))
# 通过交叉验证的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化
percentiles = range(1, 100, 2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    scores = cross_val_score(dt, x_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print(results)
opt = np.where(results == results.max())
# 找到体现最佳性能的特征筛选百分比
print('Optimal number of feature %d' % percentiles[opt[0][0]])
print(opt[0][0])
pl.plot(percentiles, results)
pl.xlabel('percentile of feature')
pl.ylabel('accuracy')
pl.show()










