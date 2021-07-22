import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

news = fetch_20newsgroups(subset='all')
print(len(news.data), '\n', news.data[0])
# 分割训练集和测试集 3:1
x_train, x_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=23)
# 将文本转化为特征向量
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
# 使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练集对模型参数进行估计
mnb.fit(x_train, y_train)
# 对测试样本进行类别预测
y_predict = mnb.predict(x_test)
# 准确性测评
print('The Accuracy of Linear Naive Bayes is', mnb.score(x_test, y_test))
# 对预测结果做详细分析
print('precision&recall&f1-score of Linear Naive:', '\n',
      classification_report(y_test, y_predict,
                            target_names=news.target_names))

# 求取两种模型预测结果的混淆矩阵
cmMnb = pd.DataFrame(confusion_matrix(y_test, y_predict),
                     index=news.target_names,
                     columns=news.target_names)
print('Naive Bayes混淆矩阵：', '\n', cmMnb)
# 设置Seaborn绘图风格默认
sns.set()
plt.figure(1)
sns.heatmap(cmMnb, annot=True, fmt="d",
            xticklabels=news.target_names,
            yticklabels=news.target_names)
plt.xlabel('predict'), plt.ylabel('true'), plt.title('Naive Bayes Model')
plt.show()








