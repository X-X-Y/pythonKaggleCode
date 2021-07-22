import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

news = fetch_20newsgroups(subset='all')
# 对前3000条新闻文本进行数据分割
x_train, x_test, y_train, y_test = train_test_split(
    news.data[:3000], news.target[:3000], test_size=0.25, random_state=23)
# 使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')),
               ('svc', SVC())])
# 需要实验的两个超参数的个数分别是4、3，共有12种参数组合模型
parameters = {'svc__gamma': np.logspace(-2, 1, 4),
              'svc__C': np.logspace(-1, 1, 3)}
# 初始化并行网格搜索，n_job=-1代表使用该计算机的全部CPU
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)
gs.fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
# 输出最佳模型在测试集上的准确性
print(gs.score(x_test, y_test))









